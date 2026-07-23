import sys
from ._version import __odps_sdk_version__
# Clear any already loaded pykmonitor modules to prevent loading incorrect versions.
try:
  for module_name, module in sorted(sys.modules.items(), key=lambda x: x[0]):
    if module_name.startswith('pykmonitor'):
      del sys.modules[module_name]
except Exception:
  pass

try:
  from .dataset.open_storage_row_reader_v2 import OpenStorageRowReaderV2 as OpenStorageRowReader
  __all__ = ['OpenStorageRowReader']
except:
  pass

