import os
import sys

import commentjson


def get_launch_names():
    json_file = open("../.vscode/launch.json", "r")
    launch_json = commentjson.load(json_file)
    configurations = launch_json["configurations"]
    return [config["name"] for config in configurations]


def main():
    launch_names = get_launch_names()
    out_f = open("sample_names.json", "w")
    print(commentjson.dumps({"names": launch_names}, indent=4), file=out_f)


if __name__ == '__main__':
    main()
