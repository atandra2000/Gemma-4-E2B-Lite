"""Doc↔code reference checker: every ``file.py:Symbol`` citation in the
documentation must resolve to a real symbol in this repo; line-number
anchors are banned. Fails CI when the docs drift from the code.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Doc modules import project packages (models/, training/, data/...).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOC_GLOBS = [
    "docs/**/*.md",
    "*.md",              # README.md, AGENTS.md, SKILLS.md
]

# `file.py:Symbol` or `file.py:Class.method` (backticked, or bare in prose)
SYMBOL_RE = re.compile(
    r"`?([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)"  # file.py:Symbol
    r"|`?([A-Za-z_][A-Za-z0-9_./-]*\.py)::([A-Za-z_][A-Za-z0-9_.]*)"  # file.py::Class.test (pytest style)
)
# Line-number anchors: file.py:123 or file.py L123 / (L123-140) / bare L123–456.
# "L2 norm", "L1 loss", etc. are math terms, not anchors — excluded.
LINE_ANCHOR_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_./-]*\.py)\s*[:L]\s*\d+"
    r"|\bL\d+(?:\s*[–-]\s*\d+)?\b(?!\s*(?:norm|regularization|reg|loss|penalty|distance|ball|error|regularizer)\b)"
)


def _load_module(path: Path):
    """Import a .py file by path (handles non-package files like scripts/)."""
    name = f"_docref_{path.stem}_{abs(hash(str(path))) % (10 ** 8)}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _has_instance_attr(cls, name: str) -> bool:
    """Instance attrs assigned via self.name = … are invisible to hasattr."""
    try:
        import inspect
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    return re.search(rf"self\.{re.escape(name)}\s*=", src) is not None


def _resolve(file_ref: str, symbol: str):
    """Resolve file.py:Symbol; symbol may be dotted (Class.method)."""
    path = (ROOT / file_ref).resolve()
    if not path.exists():
        return False, f"{file_ref} does not exist"
    try:
        module = _load_module(path)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        return False, f"{file_ref} failed to import: {type(exc).__name__}: {exc}"
    obj = module
    for part in symbol.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if isinstance(obj, type) and _has_instance_attr(obj, part):
            continue
        return False, f"{file_ref}:{symbol} — {part!r} not found"
    return True, None


def _doc_files():
    found = []
    for pattern in DOC_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and p.suffix == ".md":
                found.append(p)
    return sorted(set(found))


def _check_doc(path: Path) -> list[str]:
    errors = []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)

    for m in SYMBOL_RE.finditer(text):
        file_ref, symbol = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
        if file_ref == "file.py":
            continue  # literal metasyntax placeholder in the writing contract
        ok, err = _resolve(file_ref, symbol)
        if not ok:
            errors.append(f"{rel}: unresolved citation `{file_ref}:{symbol}` — {err}")

    for m in LINE_ANCHOR_RE.finditer(text):
        errors.append(f"{rel}: line-number anchor banned: {m.group(0)!r}")
    return errors


def _check_links(path: Path) -> list[str]:
    """Validate intra-repo markdown links; code fences are stripped so
    source snippets are not scanned. Links may be doc-relative or root-relative."""
    LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
    errors = []
    text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT)
    for m in LINK_RE.finditer(text):
        target = m.group(1).strip()
        if not target or target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        candidates = [(path.parent / path_part).resolve(), (ROOT / path_part).resolve()]
        if not any(c.exists() for c in candidates):
            errors.append(f"{rel}: broken link -> {target}")
    return errors


def _check_snippets(path: Path) -> list[str]:
    """Every non-trivial python code block must either be marked
    `# illustrative` or carry a `# verified` marker; unmarked blocks are
    flagged so readers know the snippet was not executed."""
    FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
    errors = []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)
    for m in FENCE_RE.finditer(text):
        block = m.group(1)
        if len(block.splitlines()) < 3:
            continue  # trivial one-liners need no marker
        if "# illustrative" in block or "# verified" in block:
            continue
        errors.append(f"{rel}: unmarked python code block (add `# illustrative` or `# verified`)")
    return errors


def test_doc_references_resolve():
    errors = []
    for path in _doc_files():
        errors.extend(_check_doc(path))
    assert not errors, "\n".join(errors)


def test_no_line_number_anchors():
    errors = []
    for path in _doc_files():
        errors.extend(_check_doc(path))
    # line anchors are reported by the same pass; this test shares the data
    anchor_errors = [e for e in errors if "line-number anchor" in e]
    assert not anchor_errors, "\n".join(anchor_errors)


def test_doc_links_resolve():
    errors = []
    for path in _doc_files():
        errors.extend(_check_links(path))
    assert not errors, "\n".join(errors)


def test_doc_snippets_marked():
    errors = []
    for path in _doc_files():
        errors.extend(_check_snippets(path))
    assert not errors, "\n".join(errors)


if __name__ == "__main__":
    all_errors = []
    for path in _doc_files():
        all_errors.extend(_check_doc(path))
        all_errors.extend(_check_links(path))
        all_errors.extend(_check_snippets(path))
    if all_errors:
        print("\n".join(all_errors))
        sys.exit(1)
    print(f"OK: {len(_doc_files())} docs, all symbol citations resolve, no line anchors")
