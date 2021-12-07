import sys
import os
import commentjson
import subprocess
from colorama import Fore, Style
import utils
from logger import Log, Status


def test():
    try:
        f = open("text.txt", "r")
    except Exception as e:
        print("FAILED: ", e, file=sys.stderr)
        return -1


def build_kernel_slicer():
    res, msg = utils.cmake_build("cmake-build-debug", "Debug", return_to_root=True)
    if res.returncode != 0:
        Log().info("kernel_slicer debug build: " + msg, status=Status.FAILED)
        Log().save_std_output("kernel_slicer_debug_build", res.stdout.decode(), res.stderr.decode())
        exit(1)
    else:
        Log().info("kernel_slicer debug build", status=Status.OK)

    res, msg = utils.cmake_build("cmake-build-release", "Release", return_to_root=True)
    if res.returncode != 0:
        Log().info("kernel_slicer release build: " + msg, status=Status.FAILED)
        Log().save_std_output("kernel_slicer_debug_build", res.stdout.decode(), res.stderr.decode())
        exit(1)
    else:
        Log().info("kernel_slicer release build", status=Status.OK)

    return 0


if __name__ == '__main__':
    workdir = os.getcwd()
    Log().set_workdir(workdir)
    if len(sys.argv) >= 2:
        try:
            workdir = os.path.join(os.getcwd(), sys.argv[1])
            os.chdir(workdir)
        except Exception as e:
            print("Wrong working dictionary path:\n", e, file=sys.stderr)
            exit(1)
    Log().print("Running in root: ", workdir)
    build_kernel_slicer()
    # test()
    Log().close()
