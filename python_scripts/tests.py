import sys
import os
import subprocess


import utils
from tests_config import TestsConfig, Backend
from sample_config import read_sample_config_list, SampleConfig, ShaderLang
from logger import Log, Status
from download_resources import download_resources
from compare_images import compare_generated_images
from compare_json import compare_generated_json_files


def compile_shaders(shader_lang):
    Log().info("Compiling {} shaders".format(shader_lang.name))
    res = None
    if shader_lang == ShaderLang.OPEN_CL:
        res = subprocess.run(["bash", "z_build.sh"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif shader_lang == ShaderLang.GLSL:
        os.chdir("shaders_generated")
        res = subprocess.run(["bash", "build.sh"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir("..")

    return res


def compile_sample(sample_config, num_threads=1):
    os.chdir(sample_config.root)
    Log().info("Building sample: {}".format(sample_config.name))
    res, msg = utils.cmake_build("build", "Release", return_to_root=True, num_threads=num_threads)
    if res.returncode != 0:
        Log().status_info("{} release build: ".format(sample_config.name) + msg, status=Status.FAILED)
        Log().save_std_output("{}_release_build".format(sample_config.name), res.stdout.decode(), res.stderr.decode())
        return -1

    res = compile_shaders(sample_config.shader_lang)
    if res.returncode != 0:
        Log().status_info("{} shaders compilation: ".format(sample_config.name), status=Status.FAILED)
        Log().save_std_output("{}_shaders".format(sample_config.name), res.stdout.decode(), res.stderr.decode())
        return -1

    return 0


def run_sample(test_name, on_gpu=False, gpu_id=0):
    Log().info("Running sample: {0}, gpu={1}".format(test_name, on_gpu))
    
    args = ["./build/testapp", "--test"]
    args = args + ["--gpu_id", str(gpu_id)]  # for single launch samples
    if on_gpu:
        args = args + ["--gpu"]

    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        Log().status_info("Failed to launch sample {0} : {1}".format(test_name, e), status=Status.FAILED)
        return -1
    if res.returncode != 0:
        Log().status_info("{}: launch".format(test_name), status=Status.FAILED)
        Log().save_std_output(test_name, res.stdout.decode(), res.stderr.decode())
        return -1

    return 0


def run_kslicer_and_compile_sample(sample_config: SampleConfig, test_config: TestsConfig, megakernel=False):
    Log().info("Generating files by kernel_slicer with params: [\n" +
               "\torig cpp file: {}\n".format(os.path.relpath(sample_config.orig_cpp_file)) +
               ("\tmegakernel: {}\n".format(megakernel) if sample_config.has_megakernel_key else "") +
               "]")

    kslicer_args = sample_config.get_kernel_slicer_args(megakernel=megakernel)
    res = subprocess.run(["./cmake-build-release/kslicer", *kslicer_args],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        Log().status_info("{}: kernel_slicer files generation".format(sample_config.name), status=Status.FAILED)
        Log().save_std_output(sample_config.name, res.stdout.decode(), res.stderr.decode())
        return -1

    return_code = compile_sample(sample_config, test_config.num_threads)
    if return_code != 0:
        return -1

    return 0


def run_test(sample_config: SampleConfig, test_config: TestsConfig):
    Log().info("Running test: {}".format(sample_config.name))
    workdir = os.getcwd()
    final_status = Status.OK

    return_code = run_kslicer_and_compile_sample(sample_config, test_config, megakernel=False)
    if return_code != 0:
        os.chdir(workdir)
        return -1

    if Backend.CPU in test_config.backends:
        return_code = run_sample(sample_config.name, on_gpu=False, gpu_id=test_config.gpu_id)
        if return_code != 0:
            os.chdir(workdir)
            return -1

    if Backend.GPU in test_config.backends:
        return_code = run_sample(sample_config.name, on_gpu=True, gpu_id=test_config.gpu_id)
        if return_code != 0:
            os.chdir(workdir)
            return -1

        final_status = Status.worst_of([
            final_status, compare_generated_images(), compare_generated_json_files()
        ])

        if sample_config.has_megakernel_key:
            os.chdir(workdir)
            return_code = run_kslicer_and_compile_sample(sample_config, test_config, megakernel=True)
            if return_code != 0:
                os.chdir(workdir)
                return -1
            return_code = run_sample(sample_config.name, on_gpu=True, gpu_id=test_config.gpu_id)
            if return_code != 0:
                os.chdir(workdir)
                return -1

            final_status = Status.worst_of([
                final_status, compare_generated_images(), compare_generated_json_files()
            ])

    final_msg = {
        Status.OK: "\"{}\" finished successfully".format(sample_config.name),
        Status.WARNING: "\"{}\" finished with warnings".format(sample_config.name),
        Status.FAILED: "\"{}\" has errors".format(sample_config.name)
    }
    Log().status_info(final_msg[final_status], status=final_status)
    os.chdir(workdir)


def tests(test_config):
    sample_configs = read_sample_config_list(test_config.sample_names_path)
    for config in sample_configs:
        run_test(config, test_config)


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


def main():
    workdir = os.getcwd()
    test_config = TestsConfig(argv=sys.argv[1:])
    try:
        workdir = os.path.abspath(test_config.workdir)
        os.chdir(workdir)
    except Exception as e:
        print("Wrong working dictionary path:\n", e, file=sys.stderr)
        exit(1)
    Log().set_workdir(workdir)
    Log().print("Running in root: ", workdir)
    download_resources()
    create_clspv_symlink("apps/clspv", "apps/tests/clspv")
    build_kernel_slicer(test_config.num_threads)
    tests(test_config)
    Log().close()


if __name__ == '__main__':
    main()
