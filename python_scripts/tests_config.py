import argparse
from enum import Enum


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', type=str, default=".", help="test script working dir path")
    parser.add_argument("--num_threads", type=int, default=8, help="number of threads for cmake build")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id for sample execution")
    parser.add_argument("--backend", type=str, default="all", choices=["all", "cpu", "gpu"],
                        help="required backend to be executed by tests")
    parser.add_argument("--sample_names", type=str, default="",
                        help="path to json file with sample names which are desired to be tested")

    args = parser.parse_args(args=argv)

    return args


class Backend(Enum):
    CPU = 0
    GPU = 1

    @staticmethod
    def get_backend_set(arg: str):
        arg = arg.upper()
        if arg == "ALL":
            return {Backend.CPU, Backend.GPU}
        else:
            return {Backend[arg]}


class TestsConfig:
    def __init__(self, argv):
        args = parse_args(argv)

        self.workdir = args.workdir
        self.num_threads = args.num_threads
        self.backends = Backend.get_backend_set(args.backend)
        self.gpu_id = args.gpu_id
        self.sample_names_path = args.sample_names

