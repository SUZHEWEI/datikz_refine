from contextlib import contextmanager
from datetime import datetime
from json import load
from urllib.request import Request, urlopen
import os

from datasets import (
    disable_progress_bar,
    enable_progress_bar,
    is_progress_bar_enabled,
)

def get_creation_time(repo):
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "User-Agent": "DaTikZ",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(f"https://api.github.com/repos/{repo}", headers=headers)
    return datetime.strptime(load(urlopen(req))['created_at'], "%Y-%m-%dT%H:%M:%SZ")

def lines_startwith(string, prefix):
    return all(line.startswith(prefix) for line in string.splitlines())

def lines_removeprefix(string, prefix):
    return "".join(line.removeprefix(prefix) for line in string.splitlines(keepends=True))

@contextmanager
def no_progress_bar():
    if is_progress_bar_enabled():
        try:
            yield disable_progress_bar()
        finally:
            enable_progress_bar()
