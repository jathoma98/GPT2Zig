"""Materialize the GPT-2 model as a build artifact inside the venv.

Downloads gpt2 into the venv-local HF cache (python/.venv/hf_cache, set by _hf_env) and points the
repo-local `models/gpt2` symlink at the resulting snapshot dir, so the rest of the build/runtime can
reference stable `models/gpt2/...` paths without touching the host home dir. Run by ensureVenvReady
in build.zig once the venv deps are installed.
"""
import os
import _hf_env  # noqa: F401  — sets HF_HOME before huggingface_hub import

from huggingface_hub import snapshot_download

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/.. == repo root
_LINK = os.path.join(_REPO, "models", "gpt2")


def main() -> None:
    snap = snapshot_download(
        "gpt2",
        allow_patterns=["config.json", "merges.txt", "model.safetensors"],
    )
    os.makedirs(os.path.dirname(_LINK), exist_ok=True)
    # Idempotent relink: drop any prior link/file/dir, then point at the fresh snapshot.
    if os.path.islink(_LINK) or os.path.isfile(_LINK):
        os.unlink(_LINK)
    elif os.path.isdir(_LINK):
        import shutil

        shutil.rmtree(_LINK)
    os.symlink(snap, _LINK)
    print(f"models/gpt2 -> {snap}")


if __name__ == "__main__":
    main()
