import sys
import os
from colorama import Fore, Style
from enum import Enum
import utils


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Status(Enum):
    OK     = 0
    FAILED = 1


class Log(metaclass=Singleton):
    def __init__(self):
        self.workdir = None
        self.log_dir = None
        self.log_file = None

    def set_workdir(self, workdir):
        self.workdir = workdir
        self.log_dir = os.path.join(self.workdir, "LOG_DIR")

        utils.clear_dir(self.log_dir)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        try:
            self.log_file = open(os.path.join(self.log_dir, "log.txt"), "w")
        except Exception as e:
            print(e, file=sys.stderr)
            exit(-1)

    def info(self, *args, status=Status.OK):
        status_color = {
            Status.OK: Fore.GREEN,
            Status.FAILED: Fore.RED
        }

        print("["+status_color[status]+status.name+Style.RESET_ALL+"] ", end="", file=sys.stdout)
        print("["+status.name+"] ", end="", file=self.log_file)
        self.print(*args)

    def print(self, *args):
        print(*args, file=sys.stdout)
        print(*args, file=self.log_file)

    def save_std_output(self, proc_name, stdout, stderr):
        out_f_name = proc_name+"__stdout.txt"
        out_f = open(os.path.join(self.log_dir, out_f_name), "w")
        print(stdout, file=out_f)
        self.print("stdout is saved to file "+out_f_name)

        err_f_name = proc_name + "__stderr.txt"
        err_f = open(os.path.join(self.log_dir, err_f_name), "w")
        print(stderr, file=err_f)
        self.print("stderr is saved to file " + err_f_name)

    def close(self):
        self.log_file.close()

