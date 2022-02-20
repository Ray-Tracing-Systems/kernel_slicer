# Python autotests

`tests.py` is the main script which performs next steps:

* downloads external resources from github repos (check `download_resources.py`).
* creates symlink in `apps/tests` dir to clspv file. This **requires clspv** compiler to be in `apps` dir.
* compiles debug and release builds of kernel slicer.
* loads launch configs from `.vscode/launch.json`. Then for each configuration:
	* runs kernel slicer codegen and compiles sample
	* runs sample with different cmd options
	* compares generated output (images and json files which names start from `zout_*`)
* saves log info and output of failed samples to `LOG_DIR` directory.

## Package requirements

All required python packages are listed in `requirements.txt`\
To install them run:
```
pip3 install -r requirements.txt
```

## Launch parameters

To see actual list of cmd params from **root** of kernel slicer project run:
```
python3 python_scripts/tests.py -h
```

Positional arguments:
* `workdir` - working directory path for autotests (actually this is relative path to kernel slicer root)

Optional arguments:
* `--num_threads NUM_THREADS` - the number of threads for cmake build exectution
* `--gpu_id GPU_ID` - GPU id for sample execution (this option is created for cases of multiple GPU on one machine (for ex. notebook can have both discrete and integrated GPUs))
* `--backend {all, cpu, gpu}` - required backend to be executed by autotests
* `--sample_names SAMPLE_NAMES` - path to json file with sample names which are desired to be tested (see section **Partial testing**)

Launch example (from kernel slicer root dir):
```
python3 python_scripts/tests.py . --backend=all --gpu_id=0 --num_threads=6
```

## Partial testing

Sometimes you don't want to run all existing test samples, but only some of them. There is a useful way to do that.

First of all, you need to generate json file which contains names of all existing sample configurations.
To do that, go to `python_scripts` directory and run:
```
python3 gen_sample_list.py
```
This will create `sample_names.json` file with names of launch configs.\
To select samples to be executed by autotests just **comment unnecessary names out**.


Finally, add path to `sample_names.json` file in optional arg `--sample_names`:
```
python3 python_scripts/tests.py . --backend=all --gpu_id=0 --num_threads=6 --sample_names=python_scripts/sample_names.json
```