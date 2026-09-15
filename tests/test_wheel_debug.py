import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "wheel_debug.py"
SPEC = importlib.util.spec_from_file_location("wheel_debug", SCRIPT)
wheel_debug = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_debug)


def minimal_elf(text=b"\x90\x90\x90\xc3", entry_size=0):
    names = b"\0.shstrtab\0.text\0"
    text_offset = 64
    names_offset = text_offset + len(text)
    section_offset = 96
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    header = struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        62,
        1,
        0,
        0,
        section_offset,
        0,
        64,
        0,
        0,
        64,
        3,
        1,
    )
    section = struct.Struct("<IIQQQQIIQQ")
    null_section = section.pack(*([0] * 10))
    string_section = section.pack(
        names.index(b".shstrtab"), 3, 0, 0, names_offset, len(names), 0, 0, 1, 0
    )
    text_section = section.pack(
        names.index(b".text"),
        1,
        0x6,
        0x1000,
        text_offset,
        len(text),
        0,
        0,
        16,
        entry_size,
    )
    prefix = ident + header + text + names
    return (
        prefix
        + b"\0" * (section_offset - len(prefix))
        + null_section
        + string_section
        + text_section
    )


class WheelDebugElfTest(unittest.TestCase):
    def test_alloc_section_content_change_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = Path(tmp) / "before.so"
            after = Path(tmp) / "after.so"
            before.write_bytes(minimal_elf())
            after.write_bytes(minimal_elf(b"\x91\x90\x90\xc3"))

            before_sections = wheel_debug._alloc_sections(before)
            after_sections = wheel_debug._alloc_sections(after)

            self.assertEqual(
                before_sections[".text#0"][-1],
                hashlib.sha256(b"\x90\x90\x90\xc3").hexdigest(),
            )
            self.assertEqual(
                wheel_debug._changed_alloc_sections(
                    before_sections, after_sections
                ),
                [".text#0"],
            )

    def test_non_elf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-elf"
            path.write_bytes(b"not an elf")
            with self.assertRaises(RuntimeError):
                wheel_debug._alloc_sections(path)

    def test_entry_size_normalization_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = Path(tmp) / "before.so"
            after = Path(tmp) / "after.so"
            before.write_bytes(minimal_elf(entry_size=0))
            after.write_bytes(minimal_elf(entry_size=8))

            self.assertEqual(
                wheel_debug._changed_alloc_sections(
                    wheel_debug._alloc_sections(before),
                    wheel_debug._alloc_sections(after),
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
