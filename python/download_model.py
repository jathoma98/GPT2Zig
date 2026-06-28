"""Materialize the GPT-2 model as a build artifact inside the venv.

Downloads gpt2 into the venv-local HF cache (python/.venv/hf_cache, set by _hf_env) and points the
repo-local `models/gpt2` link at the resulting snapshot dir, so the rest of the build/runtime can
reference stable `models/gpt2/...` paths without touching the host home dir. The link is a symlink on
POSIX and a directory junction on Windows — a junction is the portable symlink equivalent there: it
needs no admin/Developer-Mode privileges (unlike os.symlink) for a local same-volume target. Run by
ensureVenvReady in build.zig once the venv deps are installed.
"""
import os
import stat
import subprocess
import _hf_env  # noqa: F401  — sets HF_HOME before huggingface_hub import

from huggingface_hub import snapshot_download

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/.. == repo root
_LINK = os.path.join(_REPO, "models", "gpt2")
_IS_WINDOWS = os.name == "nt"


def _is_junction(path: str) -> bool:
    # A junction reports as a directory (isdir True) but is a reparse point that os.path.islink does
    # NOT catch. Detect it so cleanup can rmdir (unlink just the junction) rather than rmtree, which
    # would recurse into and delete the cached snapshot the junction points at.
    isjunction = getattr(os.path, "isjunction", None)  # os.path.isjunction is 3.12+
    if isjunction is not None:
        return isjunction(path)
    try:
        attrs = os.lstat(path).st_file_attributes  # Windows-only field
    except (AttributeError, OSError):
        return False
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _drop_existing(path: str) -> None:
    # Idempotent relink: clear any prior link/file/dir at the path. Order matters — the junction
    # check must precede the plain-dir branch, since a junction also satisfies isdir.
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif _is_junction(path):
        os.rmdir(path)
    elif os.path.isdir(path):
        import shutil

        shutil.rmtree(path)


def _link(target: str, link: str) -> None:
    if _IS_WINDOWS:
        # `mklink /J` creates a directory junction. It's a cmd builtin (hence `cmd /c`) and takes
        # an absolute local target, which snapshot_download returns. No privileges required.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, target],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        os.symlink(target, link)


def main() -> None:
    snap = snapshot_download(
        "gpt2",
        allow_patterns=["config.json", "merges.txt", "model.safetensors"],
    )
    os.makedirs(os.path.dirname(_LINK), exist_ok=True)
    _drop_existing(_LINK)
    _link(snap, _LINK)
    print(f"models/gpt2 -> {snap}")


if __name__ == "__main__":
    main()
