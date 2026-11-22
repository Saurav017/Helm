"""
helm/protocols/filesystem_adapter.py — Robust filesystem resolver and tool adapter.

Provides case-insensitive file resolution for read_file and graceful fallback
to project root for list_directory when passed invalid paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

# Preferred file extensions in resolution priority order
_PREFERRED_EXTENSIONS = (
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".rs",
    ".go",
)


def resolve_file_path(path: str | Path, base_dir: Path | str = ".") -> Path | None:
    """
    Resolve a file path, checking exact path first, then falling back to
    case-insensitive and extension-matched files in the same directory.

    Example:
        'readme' -> Path('README.md')
        'README.MD' -> Path('README.md')
    """
    if not path:
        return None

    raw_path = Path(path)
    base = Path(base_dir)

    # 1. Determine directory to search
    parent_dir = raw_path.parent
    search_dirs: list[Path] = []
    if parent_dir == Path() or str(parent_dir) in ("", "."):
        search_dirs = [base, Path()]
    else:
        search_dirs = [base / parent_dir, parent_dir]

    target_name = raw_path.name.lower()
    target_stem = raw_path.stem.lower()

    for s_dir in search_dirs:
        try:
            if not s_dir.is_dir():
                continue
            entries = list(s_dir.iterdir())
        except OSError:
            continue

        # Pass 0: exact filename match (case-sensitive)
        for entry in entries:
            try:
                if entry.is_file() and entry.name == raw_path.name:
                    return entry
            except OSError:
                continue

        # Pass 1: exact filename match (case-insensitive, e.g. readme.md -> README.md)
        for entry in entries:
            try:
                if entry.is_file() and entry.name.lower() == target_name:
                    return entry
            except OSError:
                continue

        # Pass 2: stem match (case-insensitive, e.g. readme -> README.md)
        stem_matches: list[Path] = []
        for entry in entries:
            try:
                if entry.is_file() and entry.stem.lower() in (target_name, target_stem):
                    stem_matches.append(entry)
            except OSError:
                continue

        if stem_matches:
            stem_matches.sort(
                key=lambda p: (
                    _PREFERRED_EXTENSIONS.index(p.suffix.lower())
                    if p.suffix.lower() in _PREFERRED_EXTENSIONS
                    else 999,
                    p.name.lower(),
                )
            )
            return stem_matches[0]

    # 2. Fallback check for absolute or direct paths outside search_dirs
    candidates = [raw_path, base / raw_path]
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            pass

    return None


def read_file_sync(path: str | Path, base_dir: Path | str = ".") -> dict[str, Any]:
    """
    Read file content, resolving case-insensitive or extension matches in the same
    directory if the exact path does not exist.
    """
    resolved = resolve_file_path(path, base_dir)
    if resolved is None or not resolved.is_file():
        err_msg = f"FileNotFoundError: [Errno 2] No such file or directory: '{path}'"
        return {
            "is_error": True,
            "error": err_msg,
            "content": [{"type": "text", "text": err_msg}],
        }

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        return {
            "is_error": False,
            "resolved_path": str(resolved),
            "content": [{"type": "text", "text": content}],
        }
    except Exception as exc:
        err_msg = f"Error reading file '{resolved}': {exc}"
        return {
            "is_error": True,
            "error": err_msg,
            "content": [{"type": "text", "text": err_msg}],
        }


async def read_file(path: str | Path, base_dir: Path | str = ".") -> dict[str, Any]:
    """
    Async implementation of read_file. Reads the file directly, resolving
    case-insensitive matches if the exact path does not exist.
    """
    return await asyncio.to_thread(read_file_sync, path, base_dir)


def format_directory_entries(target_dir: Path) -> str:
    """Format directory entries into standard MCP filesystem text format."""
    lines: list[str] = []
    try:
        entries = sorted(target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for item in entries:
            kind = "[DIR]" if item.is_dir() else "[FILE]"
            try:
                size_str = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            except OSError:
                size_str = ""
            lines.append(f"{kind} {item.name}{size_str}")
    except OSError as exc:
        lines.append(f"Error accessing directory '{target_dir}': {exc}")
    return "\n".join(lines)


def list_directory_sync(
    path: str | Path = ".",
    base_dir: Path | str = ".",
) -> dict[str, Any]:
    """
    List directory entries. If path is invalid or does not exist on disk,
    falls back gracefully to listing the project root '.' alongside a mild warning.
    """
    p = Path(path) if path else Path()
    base = Path(base_dir)

    target_dir: Path
    warning: str | None = None

    is_valid_dir = False
    try:
        if p.is_dir():
            target_dir = p
            is_valid_dir = True
        elif (base / p).is_dir():
            target_dir = base / p
            is_valid_dir = True
    except OSError:
        is_valid_dir = False

    if not is_valid_dir:
        target_dir = base if base.is_dir() else Path()
        warning = f"Path '{path}' not found, listed root directory instead"

    listing_text = format_directory_entries(target_dir)
    final_text = f"{warning}\n\n{listing_text}" if warning else listing_text

    result: dict[str, Any] = {
        "is_error": False,
        "content": [{"type": "text", "text": final_text}],
        "path": str(target_dir),
    }
    if warning:
        result["warning"] = warning
    return result


async def list_directory(
    path: str | Path = ".",
    base_dir: Path | str = ".",
) -> dict[str, Any]:
    """
    Async implementation of list_directory. Falls back to '.' with a mild warning
    if the path is invalid or does not exist.
    """
    return await asyncio.to_thread(list_directory_sync, path, base_dir)
