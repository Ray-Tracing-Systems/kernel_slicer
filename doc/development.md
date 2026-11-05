# Tests and repository layout

## Tests

Tests are in `apps/tests/` (one feature per test) and the test applications in `apps/`. Each folder
has `ktest.json` with the list of variants (back-end, `kmake.json` config, options). Build kslicer,
then run from the repository root (Python 3 standard library only, Linux and Windows):

```bash
python3 ktest.py                          # menu: back-ends, tests or applications, number of jobs
python3 ktest.py -b slang,glsl -s all     # without questions
python3 ktest.py --rerun-failed --write-launch   # rerun failed ones, write .vscode/launch_failed.json
```

For every variant it prints `[generation|build|comparison with CPU]` statuses, for example
`[ OK | OK | OK ] test_023_prefix_summ/slang`. Folders are processed in parallel and build folders
are reused, so a repeated run is fast. See [python_scripts/ktest/README.md](../python_scripts/ktest/README.md).

The old test runner is still available. It reads translator command lines from `.vscode/launch.json`:

```bash
cp .vscode/launch_slang.json .vscode/launch.json      # choose the set of configurations
cd python_scripts
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt
python3 gen_sample_list.py                            # writes sample_names.json; edit it to choose tests
bash run_tests.bat
```

For each configuration the runner translates the sample, compiles the shaders and the application,
runs CPU and GPU versions and compares their `zout_*` outputs. Logs of failed tests are written to
`LOG_DIR/`. See [python_scripts/README.md](../python_scripts/README.md) for details.

## Repository layout

| Path | Contents |
|------|----------|
| `*.cpp`, `*.h` in the root | the translator: AST analysis, kernel and host code generation |
| `shaders_<backend>.cpp`, `tr_host_<backend>.cpp` | shader and host code generators of each back end |
| `templates/` | Vulkan host code templates ([inja](https://github.com/pantor/inja)) |
| `templates_slang/`, `templates_glsl/`, `templates_cuda/`, `templates_ispc/`, `templates_wk/` | shader and back end templates |
| `TINYSTL/` | minimal standard library used for parsing |
| `apps/` | samples; `apps/tests/` feature tests |
| `apps/LiteMath`, `apps/vkutils`, `apps/volk`, `apps/LiteScene`, `apps/RayTracing` | submodules: math library, Vulkan helpers, loader, scene loading, ray tracing interface |
| `ktest.py`, `python_scripts/ktest/` | test system; `python_scripts/tests.py` is the old test runner |
| `doc/` | documentation, older tutorials and [internal documentation](internals.md) |

Instructions for agents and developers who change the translator itself (build, tests, code map,
rules) are in [AGENTS.md](../AGENTS.md).
