"""Deploy ManualMate to its Hugging Face Space (Gradio SDK, ZeroGPU).

What goes to the Space (and nothing else):
  space/app.py, space/README.md, space/requirements.txt  -> Space root
  src/manualmate/                                         -> src/manualmate/
  configs/                                                -> configs/
No .env, data/, eval/ or notebooks. The repo's requirements.txt (with RAGAS
etc.) is NOT used: the Space installs space/requirements.txt.

Run on the laptop (after `hf auth login`):
    & "D:\\Projects\\manualmate\\.conda\\python.exe" scripts/deploy_space.py
    ... --create        also create the Space (public, ZeroGPU) if it does not exist
    ... --dry-run       only list the files that would be uploaded
Secrets (DATABASE_URL, GEMINI_API_KEY) are set in the Space settings, not here.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "furqan2131/manualmate"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def stage(target: Path) -> list[str]:
    for f in ("app.py", "README.md", "requirements.txt"):
        shutil.copy2(ROOT / "space" / f, target / f)
    shutil.copytree(ROOT / "src" / "manualmate", target / "src" / "manualmate", ignore=IGNORE)
    shutil.copytree(ROOT / "configs", target / "configs", ignore=IGNORE)
    return sorted(str(p.relative_to(target)).replace("\\", "/")
                  for p in target.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="create the Space if missing")
    parser.add_argument("--dry-run", action="store_true", help="list files, upload nothing")
    args = parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"{len(files)} files to upload (from git {commit}):")
        for f in files:
            if not f.startswith("src/"):
                print("  ", f)
        print(f"   + {sum(f.startswith('src/') for f in files)} files under src/")
        if args.dry_run:
            return

        from huggingface_hub import HfApi

        api = HfApi()
        if args.create:
            api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio",
                            space_hardware="zero-a10g", private=False, exist_ok=True)
            print(f"Space ready: {SPACE_ID}")
        info = api.upload_folder(
            repo_id=SPACE_ID, repo_type="space", folder_path=tmp,
            commit_message=f"Deploy ManualMate (git {commit})",
            delete_patterns=["src/**", "configs/**"],      # drop files removed locally
        )
        print(f"Uploaded: {info}")
        print(f"Space: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
