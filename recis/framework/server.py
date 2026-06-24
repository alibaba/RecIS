import json
import os
import socket
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from typing import Optional

import torch
import torch.distributed as dist
from column_io.dataset.log_util import logger

from recis.framework.request_adapter import RequestAdapter, adapt_request_payload
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
    ip = socket.gethostbyname(socket.gethostname())
    logger.info(f"Starting httpd server on {ip}:{port}...")
    httpd.serve_forever()


def server(
    orc_path,
    model,
    name_list,
    input_dataset,
    request_adapter: Optional[RequestAdapter] = None,
    need_flatten=False,
):
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
            final_data = adapt_request_payload(final_data, request_adapter)
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
            rst["data"] = extractor.extract(input_dict, need_flatten=need_flatten)
            rst["success"] = True
            if dist.get_rank() == 0:
                result_queue.put(rst)
        except Exception as e:
            rst["data"] = e
            rst["success"] = False
            print(f"catch error: {rst}")
            if dist.get_rank() == 0:
                result_queue.put(rst)


def to_jsonable(obj):
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    return obj


def create_handler():
    def parse_multipart_file_uploads(headers, rfile):
        content_length = int(headers.get("Content-Length", 0))
        content_type = headers.get("Content-Type", "")
        body = rfile.read(content_length)
        raw_message = (
            b"Content-Type: "
            + content_type.encode("latin-1")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        message = BytesParser(policy=policy.default).parsebytes(raw_message)

        file_uploads = []
        for part in message.walk():
            if part.is_multipart() or not part.get_filename():
                continue
            file_uploads.append(part.get_payload(decode=True) or b"")
        return file_uploads

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
                    b"Only multipart/form-data is supported for file uploads."
                )
                return
            file_uploads = parse_multipart_file_uploads(self.headers, self.rfile)
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            for file_data in file_uploads:
                task_queue.put(file_data)
            logger.info("[Rank 0 HTTP] Waiting for distributed processing result...")
            outputs = result_queue.get()  # 这会阻塞，直到主进程放入结果
            logger.info(f"outputs: {outputs}")
            outputs = to_jsonable(outputs)
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

    def deep_clone(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        elif isinstance(obj, dict):
            return {k: self.deep_clone(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.deep_clone(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.deep_clone(v) for v in obj)
        elif isinstance(obj, RaggedTensor):
            return {
                "values": self.deep_clone(obj.values()),
                "offsets": self.deep_clone(obj.offsets()),
                "weight": self.deep_clone(obj.weight()),
                "dense_shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
        else:
            return obj

    def _get_hook(self, name: str):
        def hook(module, input, output):
            # 将这个模块的输出保存下来，用模块名作为 key
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach().clone()
            else:
                self.features[name] = self.deep_clone(output)

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
        import numpy as np

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().tolist()

        elif isinstance(tensor, RaggedTensor):
            return {
                "values": self.tensor_to_json_list(tensor.values()),
                "offsets": self.tensor_to_json_list(tensor.offsets()),
                "weight": self.tensor_to_json_list(tensor.weight()),
                "dense_shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }

        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()

        elif isinstance(tensor, bytes):
            return tensor.decode("utf-8", errors="ignore")

        elif isinstance(tensor, (list, tuple)):
            return [self.tensor_to_json_list(v) for v in tensor]

        elif isinstance(tensor, dict):
            return {k: self.tensor_to_json_list(v) for k, v in tensor.items()}

        elif isinstance(tensor, np.generic):
            return tensor.item()

        else:
            return tensor

    def flatten(self, final_model_output, final_features):
        score_dict = None
        if isinstance(final_model_output, list) and final_model_output:
            for item in final_model_output:
                if isinstance(item, dict):
                    score_dict = item
                    break
        elif isinstance(final_model_output, dict):
            score_dict = final_model_output

        data = {}
        if isinstance(score_dict, dict):
            data.update(score_dict)

        data.update(final_features)

        return data

    def extract(self, *args, **kwargs):
        need_flatten = kwargs.pop("need_flatten", False)
        self.features = {}
        out = self.forward_func(*args, **kwargs)

        # 归一化 out -> (score, emb_or_none)
        if isinstance(out, (tuple, list)):
            if len(out) == 2:
                score, emb = out
            elif len(out) == 1:
                (score,) = out
                emb = None
            else:
                raise ValueError(
                    "forward_func must return (score, emb) or (score,) or score"
                )
        else:
            # 直接返回 score
            score, emb = out, None

        score = self.tensor_to_json_list(score)
        emb = None if emb is None else self.tensor_to_json_list(emb)

        final_features = {}
        for k, v in self.features.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    final_features[kk] = self.tensor_to_json_list(vv)
            else:
                final_features[k] = self.tensor_to_json_list(v)

        if need_flatten:
            return self.flatten(score, final_features)

        return {"score": score, "embedding": final_features, "agg_embedding": emb}
