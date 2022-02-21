import commentjson

import utils
from logger import Log, Status

float_err_threshold  = 1e-4
float_err_threshold2 = 1e-2


def float_err_status(var_name, err, err_text=""):
    if err > float_err_threshold2:
        Log().status_info("\"{0}\": {1} float err = {2} is higher than threshold".format(var_name, err_text, err), status=Status.FAILED)
        return Status.FAILED
    elif err > float_err_threshold:
        Log().status_info("\"{0}\": {1} float err = {2} is higher than threshold".format(var_name, err_text, err), status=Status.WARNING)
        return Status.WARNING
    return Status.OK


def compare_values(name, arg1, arg2):
    if type(arg1) is float:
        err = abs(arg1 - arg2)
        return float_err_status(name, err)
    elif (type(arg1) is list) and (len(arg1) > 0) and (type(arg1[0]) is float):
        err = [abs(x1 - x2) for x1, x2 in zip(arg1, arg2)]
        arr_errors = [
            (min(err), "min"),
            (sum(err)/len(err), "mean"),
            (max(err), "max")
        ]
        arr_err_status = Status.OK
        for arr_err, err_name in arr_errors:
            arr_err_status = float_err_status(name, arr_err, err_name)
            if arr_err_status is not Status.OK:
                break

        return arr_err_status
    elif arg1 != arg2:
        Log().status_info("\"{0}\": values are not same".format(name), status=Status.FAILED)
        return Status.FAILED

    return Status.OK


def compare_json_files(json1, json2):
    sim_status = Status.OK  # similarity status

    if json1.keys() != json2.keys():
        sim_status = Status.FAILED
        Log().status_info("Json files have different count of saved variables", status=sim_status)

    for key in json1.keys():
        if key not in json2:
            continue
        sim_status = Status.worst_of([sim_status, compare_values(key, json1[key], json2[key])])

    return sim_status


def compare_generated_json_files():
    Log().info("Comparing json files")
    cpu_json_file = utils.try_open("zout_cpu.json", "r")
    gpu_json_file = utils.try_open("zout_gpu.json", "r")
    if (cpu_json_file is None) and (gpu_json_file is None):
        return Status.OK
    if cpu_json_file is None:
        Log().error("Cpu json out file is missing")
        return Status.FAILED
    if gpu_json_file is None:
        Log().error("Gpu json out file is missing")
        return Status.FAILED

    try:
        cpu_json = commentjson.load(cpu_json_file)
    except Exception as e:
        Log().error("Failed to load cpu_json : {}".format(e))
        return Status.FAILED
    try:
        gpu_json = commentjson.load(gpu_json_file)
    except Exception as e:
        Log().error("Failed to load gpu_json : {}".format(e))
        return Status.FAILED

    return compare_json_files(cpu_json, gpu_json)
