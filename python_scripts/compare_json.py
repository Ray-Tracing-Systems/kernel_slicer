import commentjson

import utils
from logger import Log, Status

float_err_threshold = 1e-4


def float_err_is_small(var_name, err, err_text=""):
    if err > float_err_threshold:
        Log().status_info("\"{0}\": {1} float err = {2} is higher than threshold".format(var_name, err_text, err),
                          status=Status.FAILED)
        return False
    return True


def same_values(name, arg1, arg2):
    if type(arg1) is float:
        err = abs(arg1 - arg2)
        if not float_err_is_small(name, err):
            return False
    elif (type(arg1) is list) and (len(arg1) > 0) and (type(arg1[0]) is float):
        err = [abs(x1 - x2) for x1, x2 in zip(arg1, arg2)]
        if not float_err_is_small(name, min(err), "min") or\
            not float_err_is_small(name, sum(err)/len(err), "mean") or \
                not float_err_is_small(name, max(err), "max"):
            return False
    elif arg1 != arg2:
        Log().status_info("\"{0}\": values are not same".format(name), status=Status.FAILED)
        return False

    return True


def compare_json_files(json1, json2):
    are_same = True

    if json1.keys() != json2.keys():
        Log().status_info("Json files have different count of saved variables", Status.FAILED)
        are_same = False

    for key in json1.keys():
        if key not in json2:
            continue
        if not same_values(key, json1[key], json2[key]):
            are_same = False

    return are_same


def compare_generated_json_files():
    Log().info("Comparing json files")
    cpu_json_file = utils.try_open("zout_cpu.json", "r")
    gpu_json_file = utils.try_open("zout_gpu.json", "r")
    if (cpu_json_file is None) and (gpu_json_file is None):
        return True
    if cpu_json_file is None:
        Log().error("Cpu json out file is missing")
        return False
    if gpu_json_file is None:
        Log().error("Gpu json out file is missing")
        return False

    try:
        cpu_json = commentjson.load(cpu_json_file)
    except Exception as e:
        Log().error("Failed to load cpu_json : {}".format(e))
        return False
    try:
        gpu_json = commentjson.load(gpu_json_file)
    except Exception as e:
        Log().error("Failed to load gpu_json : {}".format(e))
        return False

    return compare_json_files(cpu_json, gpu_json)
