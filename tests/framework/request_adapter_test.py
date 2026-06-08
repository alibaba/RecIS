import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_request_adapter_module():
    module_path = REPO_ROOT / "recis" / "framework" / "request_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "request_adapter_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_def(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"function {name} not found in {path}")


def test_adapt_request_payload_applies_adapter():
    request_adapter = _load_request_adapter_module()
    payload = {"fg_record": {"poi_id": [["B001"]]}}

    adapted = request_adapter.adapt_request_payload(
        payload,
        lambda data: {
            "table_schema": {"poi_id": {"type": "ARRAY<STRING>"}},
            "fg_record": data["fg_record"],
        },
    )

    assert adapted == {
        "table_schema": {"poi_id": {"type": "ARRAY<STRING>"}},
        "fg_record": {"poi_id": [["B001"]]},
    }


def test_adapt_request_payload_preserves_payload_without_adapter():
    request_adapter = _load_request_adapter_module()
    payload = {"table_schema": {}, "fg_record": {}}

    assert request_adapter.adapt_request_payload(payload) is payload


def test_framework_server_accepts_request_adapter():
    server_path = REPO_ROOT / "recis" / "framework" / "server.py"
    server_def = _function_def(server_path, "server")
    arg_names = [arg.arg for arg in server_def.args.args]

    assert "request_adapter" in arg_names
    assert (
        "adapt_request_payload(final_data, request_adapter)" in server_path.read_text()
    )


def test_trainer_server_forwards_request_adapter():
    trainer_path = REPO_ROOT / "recis" / "framework" / "trainer.py"
    trainer_server_def = _function_def(trainer_path, "server")
    arg_names = [arg.arg for arg in trainer_server_def.args.args]
    trainer_source = trainer_path.read_text()

    assert "request_adapter" in arg_names
    assert "request_adapter=request_adapter" in trainer_source
