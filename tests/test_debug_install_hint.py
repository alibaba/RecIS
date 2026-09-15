"""Exercise splitting, RECORD regeneration and installed metadata discovery."""

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlparse


SPEC = importlib.util.spec_from_file_location(
    "wheel_debug", Path(__file__).parents[1] / "wheel_debug.py"
)
wheel_debug = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_debug)


class DebugInstallHintTest(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "linux"
        and platform.machine() == "x86_64"
        and shutil.which("gcc"),
        "requires Linux x86_64 and gcc for the ELF fixture",
    )
    def test_installed_hint_matches_actual_companion(self):
        for name, base_url in (("recis", "https://packages.example.test/packages"), ("column_io", "https://packages.example.test/packages"), ("recis", "")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source"
                package = source / name
                package.mkdir(parents=True)
                cc = root / "sample.c"
                cc.write_text("int sample(int x) { return x + 1; }\n")
                subprocess.run(
                    [
                        "gcc",
                        "-g",
                        "-shared",
                        "-fPIC",
                        str(cc),
                        "-o",
                        str(package / "sample.so"),
                    ],
                    check=True,
                )
                version = "1.2.15+cu128.torch260cu128.git12345678"
                info = source / f"{name}-{version}.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                )
                python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                (info / "WHEEL").write_text(
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                    f"Tag: {python_tag}-{python_tag}-linux_x86_64\n"
                )
                wheel_debug._wheel_command("pack", source, "-d", root)
                main = next(root.glob("*.whl"))
                with mock.patch.dict(os.environ, {"DEBUGINFO_PACKAGES_URL": base_url}):
                    debug = wheel_debug.split_wheel(main, root / "debug")
                with zipfile.ZipFile(main) as archive:
                    hint_path = f"{info.name}/DEBUGINFO.txt"
                    hint = archive.read(hint_path).decode()
                    self.assertIn(
                        hint_path, archive.read(f"{info.name}/RECORD").decode()
                    )
                if not base_url:
                    self.assertNotIn("PyPI URL:", hint)
                    self.assertIn("not configured", hint)
                    self.assertIn(f'python -m pip install "{debug.name}"', hint)
                    continue
                url = next(
                    line.removeprefix("PyPI URL: ")
                    for line in hint.splitlines()
                    if line.startswith("PyPI URL: ")
                )
                self.assertEqual(
                    unquote(urlparse(url).path),
                    f"/packages/{name.replace('_', '-')}-debuginfo/{version}/{debug.name}",
                )
                self.assertIn(f'python -m pip install "{url}"', hint)
                self.assertIn(hashlib.sha256(debug.read_bytes()).hexdigest(), hint)
                target = root / "installed"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-compile",
                        "--target",
                        str(target),
                        str(main),
                    ],
                    check=True,
                )
                distribution = next(
                    importlib.metadata.distributions(path=[str(target)])
                )
                self.assertEqual(distribution.read_text("DEBUGINFO.txt"), hint)


if __name__ == "__main__":
    unittest.main()
