"""Split ELF debug information into a companion wheel.

ELFs without DWARF are kept byte-for-byte unchanged so their symbol table stays
available to in-process symbolizers even when no companion debug file is useful.
Owned ELFs use ``--strip-debug`` to retain ``.symtab``; designated downloaded
ELFs use ``--strip-unneeded`` and fall back to the original bytes on tool failure.
Explicitly preserved libraries are never inspected or modified.
Every strip result must preserve each SHF_ALLOC section's type, flags, virtual
address, size, alignment, and content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from email.parser import Parser
from pathlib import Path
from urllib.parse import quote


def _run(*args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32(path):
    checksum = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _is_elf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def _read_exact_at(stream, offset, size, path):
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError(
            f"truncated ELF data in {path}: offset={offset}, size={size}"
        )
    return data


def _hash_region(stream, offset, size, path):
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError(
                f"truncated ELF section in {path}: offset={offset}, size={size}"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _alloc_sections(path):
    """Return stable signatures for every SHF_ALLOC section in an ELF."""
    path = Path(path)
    with open(path, "rb") as stream:
        ident = _read_exact_at(stream, 0, 16, path)
        if ident[:4] != b"\x7fELF":
            raise RuntimeError(f"not an ELF file: {path}")
        if ident[4] == 1:
            header_format = "HHIIIIIHHHHHH"
            section_format = "IIIIIIIIII"
        elif ident[4] == 2:
            header_format = "HHIQQQIHHHHHH"
            section_format = "IIQQQQIIQQ"
        else:
            raise RuntimeError(f"unsupported ELF class {ident[4]} in {path}")
        if ident[5] == 1:
            endian = "<"
        elif ident[5] == 2:
            endian = ">"
        else:
            raise RuntimeError(f"unsupported ELF byte order {ident[5]} in {path}")

        header = struct.Struct(endian + header_format)
        section = struct.Struct(endian + section_format)
        values = header.unpack(_read_exact_at(stream, 16, header.size, path))
        section_offset = values[5]
        section_entry_size = values[10]
        section_count = values[11]
        string_table_index = values[12]
        if not section_offset or section_entry_size < section.size:
            raise RuntimeError(f"ELF has no usable section table: {path}")

        def read_section(index):
            raw = _read_exact_at(
                stream,
                section_offset + index * section_entry_size,
                section.size,
                path,
            )
            return section.unpack(raw)

        section_zero = read_section(0)
        if section_count == 0:
            section_count = section_zero[5]
        if string_table_index == 0xFFFF:
            string_table_index = section_zero[6]
        if not section_count or string_table_index >= section_count:
            raise RuntimeError(f"invalid ELF section indexes in {path}")

        headers = [read_section(index) for index in range(section_count)]
        string_header = headers[string_table_index]
        string_data = _read_exact_at(
            stream, string_header[4], string_header[5], path
        )
        occurrences = {}
        signatures = {}
        for item in headers:
            name_offset, section_type, flags, address, offset, size = item[:6]
            if not flags & 0x2:  # SHF_ALLOC
                continue
            if name_offset >= len(string_data):
                raise RuntimeError(f"invalid ELF section name offset in {path}")
            name_end = string_data.find(b"\0", name_offset)
            if name_end < 0:
                raise RuntimeError(f"unterminated ELF section name in {path}")
            name = string_data[name_offset:name_end].decode(
                "utf-8", "backslashreplace"
            )
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            key = f"{name}#{occurrence}"
            content_sha256 = (
                None
                if section_type == 8  # SHT_NOBITS
                else _hash_region(stream, offset, size, path)
            )
            # sh_entsize is not loader-visible section content. GNU binutils
            # may canonicalize an LLVM-produced SHT_INIT_ARRAY value from 0
            # to the pointer width while leaving every checked property below
            # unchanged.
            signatures[key] = (
                section_type,
                flags,
                address,
                size,
                item[8],  # sh_addralign
                content_sha256,
            )
        return signatures


def _changed_alloc_sections(before, after):
    return sorted(
        set(before) ^ set(after)
        | {name for name in set(before) & set(after) if before[name] != after[name]}
    )


def _sections(path):
    return _run(os.getenv("READELF", "readelf"), "-SW", str(path))


def _has_debug_info(sections):
    return re.search(r"\.(?:debug_|zdebug_|stab)", sections) is not None


def _build_id(path):
    output = _run(os.getenv("READELF", "readelf"), "-n", str(path))
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    return match.group(1) if match else None


def _metadata(root):
    dist_infos = list(root.glob("*.dist-info"))
    if len(dist_infos) != 1:
        raise RuntimeError(f"expected one dist-info directory in {root}")
    text = (dist_infos[0] / "METADATA").read_text()
    metadata = Parser().parsestr(text)
    return dist_infos[0], metadata["Name"], metadata["Version"]


def _wheel_command(command, *args):
    subprocess.check_call([sys.executable, "-m", "wheel", command, *map(str, args)])


def _install_paths(relative, debug_data_prefix):
    # GDB's sibling .debug convention also keeps debug-only ELF files out of
    # RecIS FSLIB's top-level plugin directory scan.
    if (
        len(relative.parts) > 2
        and relative.parts[0].endswith(".data")
        and relative.parts[1] in {"purelib", "platlib"}
    ):
        installed = Path(*relative.parts[2:])
        debug_installed = installed.parent / ".debug" / f"{installed.name}.debug"
        debug_archive = debug_data_prefix / relative.parts[1] / debug_installed
        return installed, debug_installed, debug_archive
    debug_installed = relative.parent / ".debug" / f"{relative.name}.debug"
    return relative, debug_installed, debug_installed


def _process_elf(path, root, debug_root, debug_data_prefix, strategy):
    relative = path.relative_to(root)
    installed, debug_installed, debug_archive = _install_paths(
        relative, debug_data_prefix
    )
    original_sha = _sha256(path)
    entry = {
        "path": installed.as_posix(),
        "wheel_path": relative.as_posix(),
        "build_id": None,
        "original_sha256": original_sha,
        "strategy": strategy,
    }
    if strategy == "preserve":
        entry["installed_sha256"] = original_sha
        return entry, None
    try:
        entry["build_id"] = _build_id(path)
        original_sections = _sections(path)
    except (OSError, subprocess.CalledProcessError) as exc:
        if strategy == "own":
            raise RuntimeError(f"failed to inspect {relative}") from exc
        warning = f"external library left unchanged: {relative}: {exc}"
        entry["strategy"] = "external-preserved"
        entry["installed_sha256"] = original_sha
        return entry, warning
    has_symtab = ".symtab" in original_sections
    entry["has_symtab"] = has_symtab
    if not _has_debug_info(original_sections):
        # There is no useful separate debug payload. In particular, keep an
        # external .symtab for Abseil/Torch in-process symbolization.
        entry["strategy"] = "no-debug-info"
        entry["installed_sha256"] = original_sha
        return entry, None

    try:
        original_alloc_sections = _alloc_sections(path)
    except (OSError, RuntimeError) as exc:
        if strategy == "own":
            raise RuntimeError(f"failed to inspect loadable sections in {relative}") from exc
        warning = f"external library left unchanged: {relative}: {exc}"
        entry["strategy"] = "external-preserved"
        entry["installed_sha256"] = original_sha
        return entry, warning

    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged_so = Path(tmp) / path.name
        staged_debug = Path(tmp) / debug_installed.name
        shutil.copy2(path, staged_so)
        try:
            _run(
                os.getenv("OBJCOPY", "objcopy"),
                "--only-keep-debug",
                str(staged_so),
                str(staged_debug),
            )
            strip_mode = (
                "--strip-unneeded" if strategy == "external" else "--strip-debug"
            )
            _run(os.getenv("STRIP", "strip"), strip_mode, str(staged_so))
            _run(
                os.getenv("OBJCOPY", "objcopy"),
                "--remove-section=.gnu_debuglink",
                f"--add-gnu-debuglink={staged_debug.name}",
                str(staged_so),
                cwd=staged_debug.parent,
            )
            changed_alloc_sections = _changed_alloc_sections(
                original_alloc_sections, _alloc_sections(staged_so)
            )
            if changed_alloc_sections:
                raise RuntimeError(
                    "strip changed SHF_ALLOC sections in "
                    f"{relative}: {changed_alloc_sections}"
                )
            # Keep .symtab for Abseil/Torch in-process symbolization. Some empty
            # CHECK stacks are caused by unwinding and are intentionally out of scope.
            if (
                strategy == "own"
                and has_symtab
                and ".symtab" not in _sections(staged_so)
            ):
                raise RuntimeError(f"strip removed .symtab from {relative}")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            if strategy == "own":
                raise RuntimeError(
                    f"failed to split debug info from {relative}"
                ) from exc
            warning = f"external library left unchanged: {relative}: {exc}"
            entry["strategy"] = "external-preserved"
            entry["installed_sha256"] = original_sha
            return entry, warning

        debug_path = debug_root / debug_archive
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_debug, debug_path)
        os.replace(staged_so, path)

    entry.update(
        {
            "debug_path": debug_installed.as_posix(),
            "debuglink_crc32": f"{_crc32(debug_path):08x}",
            "debug_sha256": _sha256(debug_path),
            "installed_sha256": _sha256(path),
        }
    )
    return entry, None


def split_wheel(wheel_path, debug_dir, external_names=(), preserve_names=()):
    """Rewrite ``wheel_path`` and create its exact-version debuginfo wheel."""
    wheel_path = Path(wheel_path).resolve()
    debug_dir = Path(debug_dir).resolve()
    external_names = set(external_names)
    preserve_names = set(preserve_names)
    warnings = []
    existing_debug_wheels = list(debug_dir.glob("*.whl")) if debug_dir.exists() else []
    if existing_debug_wheels:
        raise RuntimeError(f"debug directory already contains wheels: {debug_dir}")

    with tempfile.TemporaryDirectory(
        prefix=".wheel-debug-", dir=wheel_path.parent
    ) as tmp:
        tmp = Path(tmp)
        unpack_dir = tmp / "unpack"
        _wheel_command("unpack", wheel_path, "-d", unpack_dir)
        roots = [path for path in unpack_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"cannot locate unpacked wheel under {unpack_dir}")
        root = roots[0]
        main_dist_info, main_name, version = _metadata(root)
        debug_name = f"{main_name}-debuginfo"
        debug_data_prefix = Path(f"{debug_name.replace('-', '_')}-{version}.data")

        debug_root = tmp / "debug-wheel"
        debug_root.mkdir()
        entries = []
        for path in sorted(
            path for path in root.rglob("*") if path.is_file() and _is_elf(path)
        ):
            if path.name in preserve_names:
                strategy = "preserve"
            elif path.name in external_names:
                strategy = "external"
            else:
                strategy = "own"
            entry, warning = _process_elf(
                path, root, debug_root, debug_data_prefix, strategy
            )
            entries.append(entry)
            if warning:
                warnings.append(warning)
                print(f"WARNING: {warning}", file=sys.stderr)

        if not any(entry.get("debug_path") for entry in entries):
            raise RuntimeError(f"no debug information found in {wheel_path.name}")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "source_distribution": main_name,
            "source_version": version,
            "source_commit": commit or None,
            "files": entries,
        }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (main_dist_info / "debug-pair.json").write_text(manifest_text)

        debug_info = debug_root / f"{debug_name.replace('-', '_')}-{version}.dist-info"
        debug_info.mkdir()
        shutil.copy2(main_dist_info / "WHEEL", debug_info / "WHEEL")
        # ``==1.2.3`` also accepts local variants; ``===`` keeps versions
        # without a local label exact (notably Column row vs. CPU-column).
        requirement = f"=={version}" if "+" in version else f"==={version}"
        (debug_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {debug_name}\n"
            f"Version: {version}\n"
            f"Summary: Debug information for {main_name}\n"
            f"Requires-Dist: {main_name} {requirement}\n"
        )
        (debug_info / "debug-pair.json").write_text(manifest_text)

        repacked_main = tmp / "repacked-main"
        repacked_debug = tmp / "repacked-debug"
        repacked_main.mkdir()
        repacked_debug.mkdir()
        _wheel_command("pack", debug_root, "-d", repacked_debug)
        packed_debug_wheels = list(repacked_debug.glob("*.whl"))
        if len(packed_debug_wheels) != 1:
            raise RuntimeError("wheel pack did not produce exactly one debug wheel")
        companion = packed_debug_wheels[0]
        distribution = re.sub(r"[-_.]+", "-", debug_name).lower()
        packages_url = os.environ.get("DEBUGINFO_PACKAGES_URL", "").rstrip("/")
        debug_url = (
            f"{packages_url}/{distribution}/{quote(version, safe='')}/"
            f"{quote(companion.name, safe='')}"
            if packages_url else None
        )
        download_hint = f"PyPI URL: {debug_url}\n\n" if debug_url else (
            "Download URL: not configured for this build.\n\n"
        )
        install_target = debug_url or companion.name
        (main_dist_info / "DEBUGINFO.txt").write_text(
            f"Debug information for {main_name} {version}\n\n"
            f"Wheel: {companion.name}\n"
            f"SHA-256: {_sha256(companion)}\n"
            + download_hint
            + "Example installation command (adjust for your environment):\n"
            + f'python -m pip install "{install_target}"\n\n'
            "Choose the interpreter and pip options (e.g. --user) to match the\n"
            "environment where the main wheel is installed.\n"
            "This URL is available only after the matching pair is published to PyPI.\n"
            "For unpublished MR/local builds, obtain this exact debug wheel from the\n"
            "same pipeline/build artifacts; do not substitute a different build.\n",
            encoding="utf-8",
        )
        _wheel_command("pack", root, "-d", repacked_main)
        packed_main_wheels = list(repacked_main.glob("*.whl"))
        if len(packed_main_wheels) != 1 or len(packed_debug_wheels) != 1:
            raise RuntimeError("wheel pack did not produce exactly one main/debug pair")

        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_wheel = debug_dir / packed_debug_wheels[0].name
        os.replace(packed_debug_wheels[0], debug_wheel)
        try:
            os.replace(packed_main_wheels[0], wheel_path)
        except Exception:
            if debug_wheel.exists():
                debug_wheel.unlink()
            raise

    if warnings and os.getenv("AONE_CI_SUMMARY_MD"):
        with open(os.environ["AONE_CI_SUMMARY_MD"], "a") as f:
            f.write("\n### Preserved external libraries\n\n")
            f.writelines(f"- {warning}\n" for warning in warnings)
    return debug_wheel
