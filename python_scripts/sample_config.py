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
        self.root = SampleConfig.__extract_sample_root(self.orig_cpp_file)
        self.shader_lang = SampleConfig.__extract_shader_lang(self.__args)
        self.has_megakernel_key = "-megakernel" in self.__args
        self.has_subgroups_key  = "-enableSubgroup" in self.__args
        self.__extract_megakernel_from_args()

    def get_kernel_slicer_args(self, megakernel=False, subgroups=False):
        out_args = self.__args
        if self.has_megakernel_key:
          out_args = out_args + ["-megakernel", "1" if megakernel else "0"]
        if self.has_subgroups_key: 
          out_args = out_args + ["-enableSubgroup", "1" if subgroups else "0"]
        return out_args

    def __extract_megakernel_from_args(self):
        if self.has_megakernel_key:
            i = self.__args.index("-megakernel")
            self.__args.pop(i)  # removes megakernel key
            self.__args.pop(i)  # removes megakernel value
        if self.has_subgroups_key:
            i = self.__args.index("-enableSubgroup")
            self.__args.pop(i)  # removes megakernel key
            self.__args.pop(i)  # removes megakernel value

    @staticmethod
    def __fix_paths_in_args(args):
        return [arg.replace("${workspaceFolder}", os.getcwd()) for arg in args]

    @staticmethod
    def __extract_sample_root(orig_cpp_file: str) -> str:
        cpp_pass = os.path.join(os.getcwd(), orig_cpp_file)
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
