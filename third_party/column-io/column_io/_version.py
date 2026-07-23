import io
import os
try:
    with io.open(os.path.join(os.path.dirname(__file__), "ODPS_SDK_VERSION"),
                 encoding="utf-8") as f:
        __odps_sdk_version__ = f.read().strip()
except (IOError, OSError):
    __odps_sdk_version__ = ""