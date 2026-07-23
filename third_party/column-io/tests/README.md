# Tests

This directory contains the test suite for `column-io`.

Tests are organized by scope so that developers can run fast checks locally and reserve heavier end-to-end validation for CI or dedicated environments.

## Layout

### `tests/unit/`

Unit tests for pure Python logic.

These tests should not depend on compiled C++ artifacts. They are intended to cover lightweight and deterministic logic such as:

- top-level package behavior in `column_io/__init__.py`
- utilities and argument validation
- pure Python code under `column_io/dataset/`
- error handling and API-level logic implemented in Python

**Properties**

- fast to run
- suitable for local development
- should work without building native extensions

---

### `tests/binding/`

Tests for the Python ↔ C++ binding layer.

These tests validate that the native library can be loaded correctly and that Python-facing bindings behave as expected. Typical coverage includes:

- dynamic library loading under `column_io/lib/`
- pybind/ctypes/cffi exposed interfaces
- Python-to-C++ argument passing
- return value conversion
- exception/error propagation across the language boundary

**Properties**

- require compiled native artifacts (for example, `.so` files)
- focus on interface correctness rather than full workflow validation

---

### `tests/integration/`

End-to-end and cross-component integration tests.

These tests exercise complete workflows and interactions across multiple layers of the project. Typical coverage includes:

- dataset read/write flows
- pipeline-level behavior
- ODPS table interaction
- Torch compilation compatibility
- multi-module integration scenarios

The old `test/` directory is moved to here, since all of them use a full-stack integration import. But it's not really integration tests.

**Properties**

- slower than unit tests
- may require additional runtime dependencies or test environments
- typically better suited for CI or pre-release validation than for every local iteration

## Guidelines

When adding new tests, place them in the narrowest applicable scope:

- use `tests/unit/` for pure Python logic
- use `tests/binding/` for native binding behavior
- use `tests/integration/` for end-to-end workflows

Please keeping this separation so that developers can preserve fast local feedback while maintaining broad coverage in CI.
