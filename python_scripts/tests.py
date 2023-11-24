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

def compile_shaders(shader_lang, shader_folder):
    Log().info("Compiling {} shaders".format(shader_lang.name))
    res = None
    if shader_lang == ShaderLang.OPEN_CL:
        res = subprocess.run(["bash", "z_build.sh"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif shader_lang == ShaderLang.GLSL:
        os.chdir(shader_folder)
        res = subprocess.run(["bash", "build.sh"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir("..")
    elif shader_lang == ShaderLang.ISPC:
        res = subprocess.run(["bash", "z_build_ispc.sh"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return res


def compile_sample(sample_config, num_threads=1, enable_ispc = False):
    os.chdir(sample_config.root)
    
    Log().info("Building sample shaders: {}".format(sample_config.name))
    res = compile_shaders(sample_config.shader_lang, sample_config.shader_folder)
    if res.returncode != 0:
        Log().status_info("{} shaders compilation: ".format(sample_config.name), status=Status.FAILED)
        Log().save_std_output("{}_shaders".format(sample_config.name), res.stdout.decode(), res.stderr.decode())
        return -1
    
    Log().info("Building sample cppcode: {}".format(sample_config.name))
    res, msg = utils.cmake_build("build", "Release", return_to_root=True, num_threads=num_threads, clearAll=True, enable_ispc=enable_ispc)
    if res.returncode != 0:
        Log().status_info("{} release build: ".format(sample_config.name) + msg, status=Status.FAILED)
        Log().save_std_output("{}_release_build".format(sample_config.name), res.stdout.decode(), res.stderr.decode())
        return -1
    
    return 0


def run_sample(test_name, on_gpu=False, gpu_id=0, on_ispc=False):
    Log().info("Running sample: {0}, gpu={1}".format(test_name, on_gpu))
    args = ["./build/testapp", "--test"]
    args = args + ["--gpu_id", str(gpu_id)]  # for single launch samples
    if on_ispc:
        args = args + ["--ispc"]
    elif on_gpu:
        args = args + ["--gpu"]
    #print("args = ", args)
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


def run_kslicer(sample_config: SampleConfig, test_config: TestsConfig, megakernel=False, subgroups=False):
    Log().info("Generating files by kernel_slicer with params: [\n" +
               "\torig cpp file: {}\n".format(os.path.relpath(sample_config.orig_cpp_file)) +
               ("\tmegakernel: {}\n".format(megakernel) if sample_config.has_megakernel_key else "") +
               ("\tsubgroups : {}\n".format(subgroups)  if sample_config.has_subgroups_key  else "") + "]")

    kslicer_args = sample_config.get_kernel_slicer_args(megakernel=megakernel, subgroups=subgroups)
    #print(kslicer_args)
    res = subprocess.run(["./cmake-build-release/kslicer", *kslicer_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        Log().status_info("{}: kernel_slicer files generation".format(sample_config.name), status=Status.FAILED)
        Log().save_std_output(sample_config.name, res.stdout.decode(), res.stderr.decode())
        return -1

    return 0


def run_test(sample_config: SampleConfig, test_config: TestsConfig, workdir):
    Log().info("Running test: {}".format(sample_config.name))
    final_status = Status.OK
    os.chdir(workdir)
    return_code = run_kslicer(sample_config, test_config, megakernel=False, subgroups=False)
    if return_code != 0:
        os.chdir(workdir)
        return -1
    return_code = compile_sample(sample_config, test_config.num_threads, enable_ispc = (sample_config.shaderType == "ispc"))
    if return_code != 0:
        return -1

    if Backend.CPU in test_config.backends:
        return_code = run_sample(sample_config.name, on_gpu=False, gpu_id=test_config.gpu_id)
        if return_code != 0:
            os.chdir(workdir)
            return -1

    if Backend.GPU in test_config.backends:
        return_code = run_sample(sample_config.name, on_gpu=True, gpu_id=test_config.gpu_id, on_ispc = (sample_config.shaderType == "ispc"))
        if return_code != 0:
            os.chdir(workdir)
            return -1

        final_status = Status.worst_of([
            final_status, compare_generated_images((sample_config.shaderType == "ispc")), compare_generated_json_files()
        ])

        if sample_config.has_megakernel_key:
            os.chdir(workdir)
            return_code = run_kslicer(sample_config, test_config, megakernel=True)
            if return_code != 0:
                os.chdir(workdir)
                return -1
            return_code = compile_sample(sample_config, test_config.num_threads, enable_ispc = False)
            if return_code != 0:
                return -1
            return_code = run_sample(sample_config.name, on_gpu=True, gpu_id=test_config.gpu_id)
            if return_code != 0:
                os.chdir(workdir)
                return -1

            final_status = Status.worst_of([
                final_status, compare_generated_images(), compare_generated_json_files()
            ])

        if sample_config.has_subgroups_key:
            os.chdir(workdir)
            return_code = run_kslicer(sample_config, test_config, megakernel=False, subgroups=True)
            if return_code != 0:
                os.chdir(workdir)
                return -1
            return_code = compile_sample(sample_config, test_config.num_threads, enable_ispc = False)
            if return_code != 0:
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
    if final_status == Status.FAILED:
        Log().info("reason: " + str([final_status, compare_generated_images(), compare_generated_json_files()]))
    os.chdir(workdir)


def tests(test_config, workdir):
    sample_configs = read_sample_config_list(test_config.sample_names_path)
    #print("sample_configs = ", sample_configs)
    for config in sample_configs:
        run_test(config, test_config, workdir)


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
    res, msg = utils.cmake_build("cmake-build-debug", "Debug", return_to_root=True, num_threads=num_threads, clearAll = False)
    if res.returncode != 0:
        Log().status_info("kernel_slicer debug build: " + msg, status=Status.FAILED)
        Log().save_std_output("kernel_slicer_debug_build", res.stdout.decode(), res.stderr.decode())
        exit(1)
    else:
        Log().status_info("kernel_slicer debug build", status=Status.OK)

    res, msg = utils.cmake_build("cmake-build-release", "Release", return_to_root=True, num_threads=num_threads, clearAll = False)
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
    #create_clspv_symlink("apps/clspv", "apps/tests/clspv")
    build_kernel_slicer(test_config.num_threads)
    tests(test_config,workdir)
    Log().close()


if __name__ == '__main__':
    main()
