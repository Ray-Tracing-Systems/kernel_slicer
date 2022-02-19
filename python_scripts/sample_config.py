import os
import commentjson
from enum import Enum
from typing import List

config_black_list = {
    "Launch (msu/vk_graphics_rt)",
    "Launch (msu/vk_graphics_rt2)"
}


class ShaderLang(Enum):
    OPEN_CL = 0
    GLSL    = 1


class SampleConfig:
    def __init__(self, name, args):
        self.name = name
        self.__args = SampleConfig.__fix_paths_in_args(args)
        self.orig_cpp_file = self.__args[0]
        self.root = SampleConfig.__extract_sample_root(self.__args)
        self.shader_lang = SampleConfig.__extract_shader_lang(self.__args)

    def get_kernel_slicer_args(self):
        return self.__args

    @staticmethod
    def __fix_paths_in_args(args):
        return [arg.replace("${workspaceFolder}", os.getcwd()) for arg in args]

    @staticmethod
    def __extract_sample_root(args):
        cpp_pass = os.path.join(os.getcwd(), args[0])
        return os.path.dirname(cpp_pass)

    @staticmethod
    def __extract_shader_lang(args):
        lang = ShaderLang.OPEN_CL

        for arg in args:
            if arg.lower() == "glsl":
                lang = ShaderLang.GLSL

        return lang


def get_sample_names(sample_names_path, configurations):
    if sample_names_path:
        sample_names_file = open(sample_names_path, "r")
        sample_names_json = commentjson.load(sample_names_file)
        return set(sample_names_json["names"])
    else:
        return {config["name"] for config in configurations}


def read_sample_config_list(sample_names_path: str) -> List[SampleConfig]:
    json_file = open(".vscode/launch.json", "r")
    launch_json = commentjson.load(json_file)
    configurations = launch_json["configurations"]
    required_samples_names = get_sample_names(sample_names_path, configurations)

    sample_configs = []

    for config in configurations:
        if config["name"] in config_black_list:
            continue
        if config["name"] not in required_samples_names:
            continue
        sample_configs.append(SampleConfig(config["name"], config["args"]))

    return sample_configs
