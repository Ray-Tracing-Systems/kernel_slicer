import sys
import os
import shutil
import commentjson
import subprocess
from skimage.io import imread
from skimage import img_as_float


def mse(img1, img2):
    return ((img1 - img2)**2).mean()


def load_and_calc_mse(img_name1, img_name2):
    img1 = img_as_float(imread(img_name1))
    img2 = img_as_float(imread(img_name2))
    return mse(img1, img2)


def get_files(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


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