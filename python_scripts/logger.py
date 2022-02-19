import sys
import os
import io
from colorama import Fore, Style
from enum import Enum
import utils


def print_to_string(*args):
    output = io.StringIO()
    print(*args, file=output, end="")
    contents = output.getvalue()
    output.close()
    return contents


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Status(Enum):
    OK      = 0
    WARNING = 1
    FAILED  = 2

    @staticmethod
    def worst_of(status_list):
        worst_status_val = max([status.value for status in status_list])
        return Status(worst_status_val)


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

    def status_info(self, *args, status=Status.OK):
        status_color = {
            Status.OK: Fore.GREEN,
            Status.WARNING: Fore.YELLOW,
            Status.FAILED: Fore.RED
        }

        print("["+status_color[status]+status.name+Style.RESET_ALL+"] ", end="", file=sys.stdout)
        print("["+status.name+"] ", end="", file=self.log_file)
        self.print(*args)

    def info(self, *args):
        self.print_colored_text(*args, color=Fore.MAGENTA)

    def error(self, *args):
        self.print_colored_text(*args, color=Fore.RED)

    def print_colored_text(self, *args, color=Fore.WHITE):
        text = print_to_string(*args)
        print(color + text + Style.RESET_ALL, file=sys.stdout)
        print(text, file=self.log_file)

    def print(self, *args):
        print(*args, file=sys.stdout)
        print(*args, file=self.log_file)

    @staticmethod
    def __convert_to_path(name):
        return name.replace("/","_")

    def save_std_output(self, proc_name, stdout, stderr):
        proc_name = self.__convert_to_path(proc_name)
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

