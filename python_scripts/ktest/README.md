# ktest: kernel_slicer test system

`ktest.py` in the repository root generates, builds and runs tests and compares GPU results with CPU
results. It uses only the Python standard library (Python 3.8+) and works on Linux and Windows.

The old test system (`python_scripts/tests.py`) still works as before and is described in
[../README.md](../README.md).

## Quick start

```bash
cd cmake-build-release && make -j16 && cd ..   # ktest does not build kslicer, build it yourself
python3 ktest.py                               # menu: choose back-ends, tests or apps, number of jobs
```

Without options that select tests (`-b`, `-s`, `--only`, `--variant`, `--external`, `--rerun-failed`,
`--list`) `ktest.py` asks a few questions (press Enter to accept the default value) and prints the
equivalent command line at the end. With them it runs without questions:

```bash
python3 ktest.py -b slang                      # all tests in apps/tests with Slang
python3 ktest.py -b slang,glsl -s all          # tests and test applications, two back-ends
python3 ktest.py -b glsl --only 23-26,45       # only tests 023..026 and 045
python3 ktest.py -s apps --only 5 --stage build  # generate and build app 05, do not run it
python3 ktest.py --list -b all -s all          # print selected runs and kslicer arguments
python3 ktest.py --help                        # all options
```

Every run prints one line:

```
  12/58 [ OK | OK | OK ] test_023_prefix_summ/slang  (6.5 s)
  13/58 [ OK |FAIL| -- ] test_042_atomic_add_float/slang_hw  (3.1 s)
            shaders: [FAILED, exit code 1]: slangc kernel1D_Test.slang ...
```

The three columns are:

1. generation: kslicer finished without errors;
2. build: shaders were compiled and the generated host code was built by cmake;
3. comparison: CPU and GPU outputs (`zout_*cpu*` and `zout_*gpu*` json and bmp files) are equal.

`OK`, `WARN` (small difference or nothing to compare), `FAIL`, `SKIP` (compiler or resources are missing)
or `--` (stage was not executed).

## Tests and test applications

* `apps/tests/NNN_name/` are **tests**: one feature per test, `-s tests` (default);
* `apps/NN_name/` are **test applications**: larger samples, `-s apps`.

Each folder contains `ktest.json` with the list of variants:

```jsonc
{
    "variants": [
        {"name": "slang",    "backend": "slang", "kmake": "kmake.json", "megakernel": true},
        {"name": "glsl",     "backend": "glsl",  "kmake": "kmake.json"},
        {"name": "slang_hw", "backend": "slang", "kmake": "kmake_hw.json", "options": {"-enableSubgroup": 0}}
    ],
    "requires": ["../resources/HydraCore"]   // optional, for apps which load scenes
}
```

* `kmake`: kernel_slicer config of the variant (relative to the folder). All kslicer options of a test
  are written in configs; several configs are used when variants differ substantially;
* `backend`: `slang`, `glsl`, `ispc` or `cuda`, passed to kslicer as `-shaderCC`;
* `options`: optional kslicer options added to the command line (they override options of the config);
* `megakernel: true`: the variant is tested twice, with `-megakernel 0` and `-megakernel 1`;
* `skip`: reason why the variant is known not to work yet (for example a feature missing in one
  back-end). Such a variant is reported as `SKIP` with the reason; `--no-skip` runs it anyway. Remove
  `skip` when the problem is fixed;
* `requires`: folders which must exist to run the sample; missing ones are cloned by `--download`.

kslicer is called from the repository root as
`kslicer <abs path>/kmake.json -shaderCC <backend> [options] [-megakernel 0|1]`; use `--list` to see the
exact command lines.

To add a test, create the folder with `test_class.h/.cpp`, `main.cpp`, `CMakeLists.txt`, `kmake.json`
and `ktest.json` (see [.claude/skills/kernel-slicer/SKILL.md](../../.claude/skills/kernel-slicer/SKILL.md)).
Nothing has to be registered elsewhere. The sample must write `zout_cpu*` files when started without
arguments and `zout_gpu*` files when started with `--gpu`.

## What happens in a run

For every variant, in the repository root:

1. the shader folder (`shaders_generated` or `shaders<suffix>`) is removed, so that `.spv` files of
   another back-end are not used;
2. kslicer is run; generation fails if the exit code is not 0, the log contains `error:`, or kslicer
   did not reach `Finished!`;
3. `build_slang.sh` / `build.sh` (`.bat` on Windows) is run in the shader folder, `z_build_ispc.sh` for ISPC;
4. the sample is configured once in `build/ktest_<backend>/` inside its folder and then rebuilt
   incrementally, so repeated runs are fast. Your own `build/` folder of the sample is not touched;
   `--clean` reconfigures from scratch;
5. `testapp --test --gpu_id N` (CPU, once per folder) and `testapp --test --gpu_id N --gpu` (`--ispc`
   for ISPC) are run in the sample folder and outputs are compared.

Variants of one folder run one after another (kslicer writes the generated files into the folder).
Different folders run in parallel (`-j`, default: half of CPU cores, at most 16). At most `--gpu-jobs`
samples (default 1) run on GPU at the same time. `--build-jobs` sets the number of compiler processes for
each build.

## Logs and failed tests

* `ktest_logs/<date_time>/<test>_<variant>.log`: all commands of a run and their output;
* `ktest_logs/<date_time>/summary.txt`: statuses and messages of all runs;
* `ktest_logs/<date_time>/launch_failed.json`: VS Code debug configurations of failed runs;
* `ktest_logs/last_failed.json`: failed runs of the last session.

```bash
python3 ktest.py --rerun-failed                # rerun only failed runs
python3 ktest.py --rerun-failed --write-launch # and write .vscode/launch_failed.json
```

`--write-launch [PATH]` writes debug configurations for failed runs (default
`.vscode/launch_failed.json`). VS Code reads only `.vscode/launch.json`: copy the configuration you need
there, or write it directly with `--write-launch .vscode/launch.json`. A file which was not written by
ktest is first renamed to `<name>.<date_time>.bak`, so your own configurations are never lost.
The configurations use `cmake-build-debug/kslicer`; edit `program` if you debug another build.

## External projects

Generation and build (without running) of a project outside of this repository:

```bash
python3 ktest.py --external ~/proj/kmake.json --cmake-dir ~/proj -b slang
python3 ktest.py --external ~/proj/ktest.json          # variants from ktest.json
python3 ktest.py --external ~/proj1 --external ~/proj2 # folders with ktest.json
```

Without `-b` the back-end is taken from `-shaderCC` of the config. The status line is `[GEN|BUILD| -- ]`.

## Options that are often needed

| option | meaning |
|---|---|
| `--kslicer PATH` | kslicer executable; default: `cmake-build-release/kslicer`, then `build/`, then `cmake-build-debug/` |
| `--stage gen\|build\|run` | last stage to execute |
| `--variant REGEX` | only variants whose name matches, e.g. `--variant _hw` |
| `--megakernel 0\|1` | only one mode of megakernel variants |
| `--no-skip` | also run variants marked with `skip` |
| `-v` | print all messages of every run, not only the first message of failed ones |
| `--build-type`, `--generator`, `--cmake-args` | how samples are built, e.g. `--generator Ninja` |
| `--gpu-id N` | GPU passed to samples |
| `--timeout S` | timeout of every command (default 1200 s) |
