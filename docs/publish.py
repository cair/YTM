import argparse
import importlib.metadata
import pathlib
import subprocess
import sys

DOCS = pathlib.Path(__file__).parent
HTML = DOCS / "_build" / "html"
BRANCH = "gh-pages"
REMOTE = "origin"
RELEASE_BRANCH = "main"
SKIP_REBASE_REMOTE = "ghp-import-skip-rebase"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref], check=False, capture_output=True).returncode == 0


def is_ancestor(ancestor: str, ref: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, ref], check=False).returncode == 0


def ghp_import(src: pathlib.Path, message: str, prefix: str | None = None) -> None:
    cmd = ["ghp-import", "-n", "-b", BRANCH, "-r", SKIP_REBASE_REMOTE, "-m", message]
    if prefix:
        cmd += ["-x", prefix]
    subprocess.run([*cmd, str(src)], check=True)


def init_pages() -> None:
    for ref in (f"refs/heads/{BRANCH}", f"refs/remotes/{REMOTE}/{BRANCH}"):
        if ref_exists(ref):
            sys.exit(f"{ref} already exists, init would wipe it")
    ghp_import(DOCS / "_pages_root", "docs: gh-pages root")


def publish() -> None:
    if not ref_exists(f"refs/heads/{BRANCH}"):
        sys.exit(f"{BRANCH} missing, run with --init first")
    remote_ref = f"refs/remotes/{REMOTE}/{BRANCH}"
    if ref_exists(remote_ref) and not is_ancestor(remote_ref, f"refs/heads/{BRANCH}"):
        sys.exit(f"{BRANCH} is behind or diverged from {REMOTE}/{BRANCH}, update it first")
    branch = git("branch", "--show-current")
    if branch != RELEASE_BRANCH:
        sys.exit(f"publish only from {RELEASE_BRANCH}, on {branch or 'detached HEAD'}")
    if git("status", "--porcelain"):
        sys.exit("working tree not clean")
    version = importlib.metadata.version("ytm")
    folder = ".".join(version.split(".")[:2])
    sha = git("rev-parse", "--short", "HEAD")
    subprocess.run(["sphinx-build", "-E", "-b", "html", str(DOCS), str(HTML)], check=True)
    ghp_import(HTML, f"docs: {version} ({sha}) in {folder}/", folder)
    ghp_import(HTML, f"docs: {version} ({sha}) in stable/", "stable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true", help=f"create {BRANCH} with the root files")
    mode.add_argument("--publish", action="store_true", help=f"build and commit docs to {BRANCH}")
    args = parser.parse_args()
    if args.init:
        init_pages()
    elif args.publish:
        publish()
    else:
        parser.print_help()
