import sys
import os
import commentjson
import subprocess
import argparse

import utils
from logger import Log, Status
from enum import Enum

config_black_list = {
    "Launch (msu/vk_graphics_rt)",
    "Launch (msu/vk_graphics_rt2)"
}


class ShaderLang(Enum):
    OPEN_CL = 0
    GLSL    = 1


def fix_paths_in_args(args):
    return [arg.replace("${workspaceFolder}", os.getcwd()) for arg in args]


# def get_main_class(args):
#     for i in range(len(args)):
#         if args[i] == "-mainClass":
#             return args[i+1]
#
#     raise RuntimeError("Can't find main class in args: {}".format(args))


def compile_shaders(shader_lang):
    Log().info("Compiling {} shaders".format(shader_lang.name))
    if shader_lang == ShaderLang.OPEN_CL:
        res = subprocess.run(["bash", "z_build.sh"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif shader_lang == ShaderLang.GLSL:
        os.chdir("shaders_generated")
        res = subprocess.run(["bash", "build.sh"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir("..")

    return res


def compile_sample(name, sample_root, shader_lang=ShaderLang.OPEN_CL, num_threads=1):
    os.chdir(sample_root)
    Log().info("Building sample: {}".format(name))
    res, msg = utils.cmake_build("build", "Release", return_to_root=True, num_threads=num_threads)
    if res.returncode != 0:
        Log().status_info("{} release build: ".format(name) + msg, status=Status.FAILED)
        Log().save_std_output("{}_release_build".format(name), res.stdout.decode(), res.stderr.decode())
        return -1

    res = compile_shaders(shader_lang)
    if res.returncode != 0:
        Log().status_info("{} shaders compilation: ".format(name), status=Status.FAILED)
        Log().save_std_output("{}_shaders".format(name), res.stdout.decode(), res.stderr.decode())
        return -1

    return 0


def extract_sample_root(args):
    cpp_pass = os.path.join(os.getcwd(), args[0])
    return os.path.dirname(cpp_pass)


def extract_shader_lang(args):
    lang = ShaderLang.OPEN_CL

    for arg in args:
        if arg.lower() == "glsl":
            lang = ShaderLang.GLSL

    return lang


def find_image_pairs():
    filenames = utils.get_files(os.getcwd())
    image_filenames = [f for f in filenames if f.startswith("zout_")]
    if len(image_filenames) % 2 != 0:
        Log().error("Odd count of generated images: it's impossible to match pairs")
        return None

    image_pairs = []
    for i in range(0, len(image_filenames), 2):
        image_pairs.append((image_filenames[i], image_filenames[i+1]))

    return image_pairs


def compare_images(img_name1, img_name2):
    mse_res = utils.load_and_calc_mse(img_name1, img_name2)
    threshold = 1e-3
    status = Status.OK if mse_res < threshold else Status.FAILED

    Log().status_info("{0}, {1} | mse = {2}".format(img_name1, img_name2, mse_res), status=status)
    return 0 if mse_res < threshold else 1


def check_generated_images(test_name):
    Log().info("Comparing images")
    image_pairs = find_image_pairs()
    if image_pairs is None:
        return -1

    res = 0
    for img1, img2 in image_pairs:
        res += compare_images(img1, img2)

    return 0 if res == 0 else -1


def run_sample(test_name, on_gpu=False, gpu_id=0):
    Log().info("Running sample: {0}, gpu={1}".format(test_name, on_gpu))
    
    args = ["./build/testapp"]
    if on_gpu:
        args = args + ["--gpu", "--gpu_id", str(gpu_id)]

    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        Log().status_info("{}: launch".format(test_name), status=Status.FAILED)
        Log().save_std_output(test_name, res.stdout.decode(), res.stderr.decode())
        return -1

    return 0


def run_test(test_name, args, num_threads=1, gpu_id=0):
    args = fix_paths_in_args(args)
    Log().info("Running test: {}".format(test_name))
    Log().info("Generating files by kernel_slicer for {}".format(args[0]))
    workdir = os.getcwd()

    res = subprocess.run(["./cmake-build-release/kslicer", *args],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        Log().status_info("{}: kernel_slicer files generation".format(test_name), status=Status.FAILED)
        Log().save_std_output(test_name, res.stdout.decode(), res.stderr.decode())
        os.chdir(workdir)
        return -1

    return_code = compile_sample(test_name, extract_sample_root(args), extract_shader_lang(args), num_threads)
    if return_code != 0:
        os.chdir(workdir)
        return -1

    return_code = run_sample(test_name, on_gpu=False)
    if return_code != 0:
        os.chdir(workdir)
        return -1
    return_code = run_sample(test_name, on_gpu=True, gpu_id=gpu_id)
    if return_code != 0:
        os.chdir(workdir)
        return -1

    return_code = check_generated_images(test_name)

    final_status = Status.OK if return_code == 0 else Status.FAILED
    os.chdir(workdir)
    Log().status_info("\"{}\" finished".format(test_name), status=final_status)


def tests(num_threads=1, gpu_id=0):
    json_file = open(".vscode/launch.json", "r")
    launch_json = commentjson.load(json_file)
    configurations = launch_json["configurations"]
    for config in configurations:
        if config["name"] in config_black_list:
            continue
        # if config["name"] != "Launch (test_004/clspv)": # @TODO: should be removed later
        #     continue
        run_test(config["name"], config["args"], num_threads, gpu_id)


def create_clspv_symlink(clspv_path, dest_path):
    Log().info("Generating clspv symlink")
    if not os.path.isfile(clspv_path):
        Log().error("Can't find clspv on path: {}".format(os.path.join(os.getcwd(), clspv_path)))
        exit(1)
    try:
        os.symlink(os.path.abspath(clspv_path), os.path.abspath(dest_path))
    except FileExistsError as e:
        Log().info("clspv symlink already exists")

    return 0


def build_kernel_slicer(num_threads):
    Log().info("Building kernel_slicer")
    res, msg = utils.cmake_build("cmake-build-debug", "Debug", return_to_root=True, num_threads=num_threads)
    if res.returncode != 0:
        Log().status_info("kernel_slicer debug build: " + msg, status=Status.FAILED)
        Log().save_std_output("kernel_slicer_debug_build", res.stdout.decode(), res.stderr.decode())
        exit(1)
    else:
        Log().status_info("kernel_slicer debug build", status=Status.OK)

    res, msg = utils.cmake_build("cmake-build-release", "Release", return_to_root=True, num_threads=num_threads)
    if res.returncode != 0:
        Log().status_info("kernel_slicer release build: " + msg, status=Status.FAILED)
        Log().save_std_output("kernel_slicer_release_build", res.stdout.decode(), res.stderr.decode())
        exit(1)
    else:
        Log().status_info("kernel_slicer release build", status=Status.OK)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', type=str, default=".", help="test script working dir path")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id for sample execution")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for cmake build")

    args = parser.parse_args()

    return args


def main():
    workdir = os.getcwd()
    args = parse_args()
    print(args)
    try:
        workdir = os.path.abspath(args.workdir)
        os.chdir(workdir)
    except Exception as e:
        print("Wrong working dictionary path:\n", e, file=sys.stderr)
        exit(1)
    Log().set_workdir(workdir)
    Log().print("Running in root: ", workdir)
    create_clspv_symlink("apps/clspv", "apps/tests/clspv")
    build_kernel_slicer(args.num_threads)
    tests(num_threads=args.num_threads, gpu_id=args.gpu_id)
    Log().close()


if __name__ == '__main__':
    main()
