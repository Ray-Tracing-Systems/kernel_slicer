import sys
import os
import shutil
import commentjson
import subprocess


def clear_dir(dir_path):
    if not os.path.isdir(dir_path):
        return

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def cmake_build(build_dir="build", build_type="Release", return_to_root=True):
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)

    os.chdir(build_dir)
    res = subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE={}".format(build_type), ".."],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return res, "cmake build failed"

    res = subprocess.run(["make", "-j", "8"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return res, "compilation failed"

    if return_to_root:
        os.chdir("..")

    return res, "build completed"
