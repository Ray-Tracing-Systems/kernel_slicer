import os
import git
from git import RemoteProgress

from logger import Log, Status


class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        max_count = max_count or 100
        percentage = int (100 * cur_count / max_count)
        if message:
            msg = "Receiving objects: {0}% {1}/{2}, {3}".format(percentage, cur_count, max_count, message)
            if cur_count == max_count:
                print(msg)
            else:
                print(msg + "\r", end="")

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)


def download_git_repo(dir_path, url, branch="master"):
    Log().info("Cloning repo {}".format(url))
    if os.path.isdir(dir_path):
        Log().status_info("Repo is already cloned", status=Status.OK)
        return

    repo = git.Repo.clone_from(url, dir_path, branch=branch, progress=CloneProgress())
    Log().status_info("Repo is cloned", status=Status.OK)


def download_resources():
    download_git_repo("apps/resources/msu-graphics-group/scenes",
                      "https://github.com/msu-graphics-group/scenes.git",
                      branch="main")
    download_git_repo("apps/resources/HydraCore",
                      "https://github.com/Ray-Tracing-Systems/HydraCore.git",
                      branch="master")

