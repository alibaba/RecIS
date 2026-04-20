import cgi
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue

import torch
import torch.distributed as dist
from column_io.dataset.log_util import logger

from recis.ragged.tensor import RaggedTensor
from recis.utils.data_utils import copy_data_to_device


task_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)


def serialize_to_tensor(data):
    return torch.tensor(list(data), dtype=torch.uint8, device="cuda")


def deserialize_from_tensor(tensor):
    data_bytes = bytes(tensor.tolist())
    return json.loads(data_bytes.decode("utf-8"))


def get_random_port(num=1):
    ports = []
    sockets = []
    try:
        for i in range(num):
            s = socket.socket()
            s.bind(("", 0))
            ports.append(s.getsockname()[1])
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()
    if len(ports) == num:
        return ports[0] if num == 1 else ports
    else:
        raise ValueError("get ports failed")


def run_http_server():
    handler_class = create_handler()
    port = get_random_port()
    server_address = ("", port)
    httpd = HTTPServer(server_address, handler_class)
    logger.info(f"Starting httpd server on port {port}...")
    httpd.serve_forever()


def server(orc_path, model, name_list, input_dataset):
    """
    Args:
        model: user define module to forward
        name_list: user define name list to hook module forward output
        input_dataset: user_define_dataset to read orc file
    """
    logger.info("offline server begin to start")
    model.eval()
    model_forward = model
    os.makedirs(orc_path, exist_ok=True)
    extractor = FeatureExtractor(model, name_list, model_forward)
    extractor.register_hooks()
    if dist.get_rank() == 0:
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()

    while True:
        rst = {}
        try:
            data_size_tensor = torch.zeros(1, dtype=torch.long, device="cuda")
            if dist.get_rank() == 0:
                logger.info("[Rank 0] Waiting for a task from HTTP thread...")
                request_data_bytes = task_queue.get()  # 从线程安全的队列获取
                data_tensor = serialize_to_tensor(request_data_bytes)
                data_size_tensor[0] = data_tensor.numel()
            # 广播数据大小 (跨机器)
            dist.broadcast(data_size_tensor, src=0)

            if dist.get_rank() != 0:
                data_tensor = torch.empty(
                    data_size_tensor[0].item(), dtype=torch.uint8, device="cuda"
                )
                # 广播实际数据 (跨机器)
            dist.broadcast(data_tensor, src=0)
            final_data = deserialize_from_tensor(data_tensor)
            logger.info(f"final_data: {final_data}")

            # 把json落到orc文件中
            from recis.framework.write_orc import write_single_sample_orc

            write_single_sample_orc(final_data, f"{orc_path}/tmp_orc.orc")
            # 创建orcdataset
            input_dataset.reset()
            iterator = iter(input_dataset)
            stop_flag, data = next(iterator)
            logger.info(f"orc dataset is {data}")

            input_dict = copy_data_to_device(data, "cuda")
            logger.info(
                f"[Rank {dist.get_rank()}] Received broadcasted task. Data size: {data_tensor.numel()}"
            )
            rst["data"] = extractor.extract(input_dict)
            rst["success"] = True
            if dist.get_rank() == 0:
                result_queue.put(rst)
        except Exception as e:
            rst["data"] = e
            rst["success"] = False
            print(f"catch error: {rst}")
            if dist.get_rank() == 0:
                result_queue.put(rst)


def create_handler():
    class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model_forward = None
            self.fg = None
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_type = self.headers["Content-Type"]
            if not content_type or "multipart/form-data" not in content_type:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(
                    "Only multipart/form-data is supported for file uploads."
                )
                return
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"}
            )
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            for field in form.keys():
                field_item = form[field]
                if (not isinstance(field_item, (tuple, list))) and field_item.filename:
                    file_data = field_item.file.read()
                    task_queue.put(file_data)
            logger.info("[Rank 0 HTTP] Waiting for distributed processing result...")
            outputs = result_queue.get()  # 这会阻塞，直到主进程放入结果
            logger.info(f"outputs: {outputs}")
            response = json.dumps(outputs)
            self.wfile.write(response.encode("utf-8"))

    return SimpleHTTPRequestHandler


class FeatureExtractor:
    def __init__(self, model, name_or_list, forward_func):
        self.model = model
        if isinstance(name_or_list, str):
            self.name_list = [name_or_list]
        else:
            self.name_list = list(name_or_list)

        self.forward_func = forward_func
        self.features = {}
        self.hooks = []

    def _get_hook(self, name: str):
        def hook(module, input, output):
            # 将这个模块的输出保存下来，用模块名作为 key
            self.features[name] = output

        return hook

    def register_hooks(self):
        print(f"--- 正在为指定模块名注册钩子(模糊匹配): {self.name_list} ---")
        targets = set(self.name_list)

        def match(name: str) -> bool:
            # 1) 完整匹配
            if name in targets:
                return True
            # 2) 尾部匹配：...sparse._embedding_engine
            for t in targets:
                if name.endswith(t):
                    return True
            return False

        for name, module in self.model.named_modules():
            print(f"current module name is {name}")
            if match(name):
                handle = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(handle)
                print(f"已为模块 '{name}' 注册钩子。")

    def remove_hooks(self):
        """移除所有已注册的钩子，防止内存泄漏"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        logger.info("--- 所有钩子已移除 ---")

    def tensor_to_json_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().tolist()

        elif isinstance(tensor, RaggedTensor):
            return {
                "values": self.tensor_to_json_list(tensor.values()),
                "offsets": self.tensor_to_json_list(tensor.offsets()),
                "weight": self.tensor_to_json_list(tensor.weight()),
                "dense_shape": list(tensor.shape),  # 或 tensor._dense_shape
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }

        elif isinstance(tensor, (list, tuple)):
            return [self.tensor_to_json_list(v) for v in tensor]

        elif isinstance(tensor, dict):
            return {k: self.tensor_to_json_list(v) for k, v in tensor.items()}

        else:
            return tensor

    def extract(self, *args, **kwargs):
        """执行模型前向传播并返回最终输出和捕获的特征"""
        # 在每次提取前清空之前的特征
        self.features = {}

        # 执行模型的前向传播
        model_output = self.forward_func(*args, **kwargs)

        final_model_output = self.tensor_to_json_list(model_output)

        final_features = {}
        for k, v in self.features.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    final_features[kk] = self.tensor_to_json_list(vv)
            else:
                final_features[k] = self.tensor_to_json_list(v)

        # 返回模型的原始输出和我们捕获的中间层特征
        output = {"score": final_model_output, "embedding": final_features}
        return output
