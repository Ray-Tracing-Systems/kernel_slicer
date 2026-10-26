"""Running of tests: generation (kslicer) -> build (shaders + cmake) -> run on CPU and GPU -> comparison.

Variants of the same folder are executed sequentially (kslicer writes generated files into the source folder),
different folders are executed in parallel. Runs on GPU are additionally limited by '--gpu-jobs'.
"""
import os
import re
import sys
import json
import time
import shutil
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .compare import OK, WARN, FAIL, compare_outputs

SKIP, NONE = 'SKIP', '--'
IS_WINDOWS = os.name == 'nt'

SHADER_SCRIPTS = {  # backend -> (script in shader folder, compiler which must be installed)
    'slang': ('build_slang.bat' if IS_WINDOWS else 'build_slang.sh', 'slangc'),
    'glsl':  ('build.bat' if IS_WINDOWS else 'build.sh', 'glslangValidator'),
}
CMAKE_FLAGS = {
    'slang': ['-DUSE_VULKAN=ON'],
    'glsl':  ['-DUSE_VULKAN=ON'],
    'ispc':  ['-DUSE_VULKAN=ON', '-DUSE_ISPC=ON'],
    'cuda':  ['-DUSE_VULKAN=OFF', '-DUSE_CUDA=ON'],
}
GPU_FLAG = {'slang': '--gpu', 'glsl': '--gpu', 'ispc': '--ispc', 'cuda': '--gpu'}
GPU_TAG  = {'slang': 'gpu', 'glsl': 'gpu', 'ispc': 'ispc', 'cuda': 'gpu'}
REQUIRED_TOOL = {'ispc': 'ispc', 'cuda': 'nvcc'}


class Context:
    def __init__(self, root, kslicer, log_dir, args):
        self.root        = root
        self.kslicer     = kslicer
        self.log_dir     = log_dir
        self.stage       = args.stage          # 'gen' | 'build' | 'run'
        self.gpu_id      = args.gpu_id
        self.build_type  = args.build_type
        self.build_jobs  = args.build_jobs
        self.cmake_args  = args.cmake_args.split() if args.cmake_args else []
        self.generator   = args.generator
        self.clean       = args.clean
        self.timeout     = args.timeout
        self.gpu_sem     = threading.Semaphore(max(1, args.gpu_jobs))
        self.print_lock  = threading.Lock()
        self.color       = sys.stdout.isatty() and not args.no_color
        self.verbose     = args.verbose
        self.no_skip     = args.no_skip
        self.cancelled   = False
        self.done        = 0
        self.total       = 0
        self.tools       = {t: shutil.which(t) for t in ('slangc', 'glslangValidator', 'ispc', 'nvcc')}


def _fmt(status, color):
    text = {OK: ' OK ', WARN: 'WARN', FAIL: 'FAIL', SKIP: 'SKIP', NONE: ' -- '}[status]
    if not color:
        return text
    code = {OK: '32', WARN: '33', FAIL: '31', SKIP: '36', NONE: '90'}[status]
    return f'\033[{code}m{text}\033[0m'


def status_line(run, color):
    st = '|'.join(_fmt(s, color) for s in run.status)
    return f'[{st}] {run.full_name}'


def _execute(cmd, cwd, log, timeout, env=None):
    """runs command, appends its output to log; returns (return code, output)"""
    log.write(f'\n$ cd {cwd}\n$ {" ".join(cmd)}\n')
    log.flush()
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
        out = p.stdout.decode('utf-8', errors='replace')
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b'').decode('utf-8', errors='replace') + f'\n[ktest]: timeout {timeout} s expired\n'
        rc = -1
    except OSError as e:
        out = f'[ktest]: can not run command: {e}\n'
        rc = -1
    log.write(out)
    log.write(f'[ktest]: exit code {rc}\n')
    log.flush()
    return rc, out


def _remove_outputs(folder, gpu_only):
    for f in os.listdir(folder):
        if f.startswith('zout_') and (not gpu_only or 'cpu' not in f):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass


def _cmake_error(out, rc):
    lines = out.splitlines()
    for l in lines:
        if 'Could NOT find' in l or 'No CMAKE_CXX_COMPILER' in l:
            return l.strip()
    for i, l in enumerate(lines):  # the message is usually on the next non empty line after 'CMake Error at ...'
        if l.startswith('CMake Error'):
            for m in lines[i + 1:i + 6]:
                if m.strip():
                    return m.strip()
            return l.strip()
    return f'exit code {rc}'


def _find_exe(build_dir, name):
    names = [name + '.exe', name] if IS_WINDOWS else [name]
    for sub in ('', 'Release', 'Debug', 'RelWithDebInfo', 'bin'):
        for n in names:
            p = os.path.join(build_dir, sub, n)
            if os.path.isfile(p):
                return p
    return None


def _shader_env(ctx):
    env = dict(os.environ)
    env.setdefault('KSLICER_SHADER_JOBS', str(ctx.build_jobs))  # generated build_slang.sh/build.sh are parallel
    return env


def generate(run, ctx, log):
    folder = run.test.folder
    if run.backend in SHADER_SCRIPTS:
        sf = run.shader_folder()
        if os.path.isdir(sf):
            shutil.rmtree(sf, ignore_errors=True)  # do not let stale .spv of other back-end be picked up
    rc, out = _execute([ctx.kslicer] + run.kslicer_args(), ctx.root, log, ctx.timeout)
    errors = [l.strip() for l in out.splitlines() if re.search(r'(^|\s)error:', l) or 'ERROR' in l]
    if rc != 0 or errors or 'Finished!' not in out:
        run.messages.append('kslicer: ' + (errors[0] if errors else f'exit code {rc}'))
        return False
    return True


def build(run, ctx, log, state):
    test = run.test
    if run.backend in SHADER_SCRIPTS:
        script, tool = SHADER_SCRIPTS[run.backend]
        if not ctx.tools[tool]:
            run.messages.append(f'{tool} is not found in PATH')
            return SKIP
        sf = run.shader_folder()
        if not os.path.isfile(os.path.join(sf, script)):
            run.messages.append(f'{os.path.relpath(os.path.join(sf, script), ctx.root)} was not generated')
            return FAIL
        cmd = ['cmd', '/c', script] if IS_WINDOWS else ['bash', script]
        rc, out = _execute(cmd, sf, log, ctx.timeout, env=_shader_env(ctx))
        if rc != 0:
            failed = [l for l in out.splitlines() if 'FAILED' in l or 'error' in l.lower()]
            run.messages.append('shaders: ' + (failed[0].strip() if failed else f'exit code {rc}'))
            return FAIL
    if run.backend in REQUIRED_TOOL and not ctx.tools[REQUIRED_TOOL[run.backend]]:
        run.messages.append(f'{REQUIRED_TOOL[run.backend]} is not found in PATH')
        return SKIP
    if run.backend == 'ispc':
        script = 'z_build_ispc.bat' if IS_WINDOWS else 'z_build_ispc.sh'
        cmd = ['cmd', '/c', script] if IS_WINDOWS else ['bash', script]
        rc, _ = _execute(cmd, test.folder, log, ctx.timeout)
        if rc != 0:
            run.messages.append(f'ispc: exit code {rc}')
            return FAIL

    build_dir = os.path.join(test.cmake_dir, 'build', 'ktest_' + run.backend)
    if (ctx.clean and build_dir not in state['cleaned']) or not os.path.isfile(os.path.join(build_dir, 'CMakeCache.txt')):
        if ctx.clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
        state['cleaned'].add(build_dir)
        os.makedirs(build_dir, exist_ok=True)
        cmd = ['cmake', '-S', test.cmake_dir, '-B', build_dir, f'-DCMAKE_BUILD_TYPE={ctx.build_type}']
        if ctx.generator:
            cmd += ['-G', ctx.generator]
        cmd += CMAKE_FLAGS[run.backend] + ctx.cmake_args
        rc, out = _execute(cmd, test.cmake_dir, log, ctx.timeout)
        if rc != 0:
            run.messages.append('cmake: ' + _cmake_error(out, rc))
            shutil.rmtree(build_dir, ignore_errors=True)  # do not keep half configured folder
            return FAIL
    cmd = ['cmake', '--build', build_dir, '--config', ctx.build_type, '--parallel', str(ctx.build_jobs)]
    rc, out = _execute(cmd, test.cmake_dir, log, ctx.timeout)
    if rc != 0:
        err = [l for l in out.splitlines() if 'error' in l.lower()]
        run.messages.append('build: ' + (err[0].strip() if err else f'exit code {rc}'))
        return FAIL
    exe = _find_exe(build_dir, test.desc.get('exe', 'testapp'))
    if exe is None:
        run.messages.append(f'executable is not found in {os.path.relpath(build_dir, ctx.root)}')
        return FAIL
    state['exe'] = exe
    return OK


def run_and_compare(run, ctx, log, state):
    test = run.test
    missing = test.missing_resources()
    if missing:
        run.messages.append('missing resources: ' + ', '.join(os.path.relpath(m, ctx.root) for m in missing) +
                            ' (use --download)')
        return SKIP
    exe, folder = state['exe'], test.folder
    base = [exe, '--test', '--gpu_id', str(ctx.gpu_id)]
    if not state.get('cpu_done'):      # CPU version does not depend on the variant: run it once per folder
        _remove_outputs(folder, gpu_only=False)
        rc, _ = _execute(base, folder, log, ctx.timeout)
        if rc != 0:
            run.messages.append(f'CPU run: exit code {rc}')
            return FAIL
        state['cpu_done'] = True
    _remove_outputs(folder, gpu_only=True)
    with ctx.gpu_sem:
        rc, _ = _execute(base + [GPU_FLAG[run.backend]], folder, log, ctx.timeout)
    if rc != 0:
        run.messages.append(f'GPU run: exit code {rc}')
        return FAIL
    cpu_files = [f for f in os.listdir(folder) if f.startswith('zout_') and 'cpu' in f]
    msgs = []
    st = compare_outputs(folder, cpu_files, GPU_TAG[run.backend], msgs)
    run.messages += msgs
    log.write('\n[ktest]: comparison: ' + st + '\n' + ''.join(f'  {m}\n' for m in msgs))
    return st


def execute_run(run, ctx, state):
    t0 = time.time()
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', run.full_name)
    run.log_file = os.path.join(ctx.log_dir, safe + '.log')
    with open(run.log_file, 'w', encoding='utf-8') as log:
        log.write(f'[ktest]: {run.full_name}\n')
        try:
            if run.skip and not ctx.no_skip:
                run.status[0] = SKIP
                run.messages.append('known problem: ' + run.skip)
            elif not generate(run, ctx, log):
                run.status[0] = FAIL
            else:
                run.status[0] = OK
                if ctx.stage != 'gen':
                    st = build(run, ctx, log, state)
                    run.status[1] = st
                    if st == OK and ctx.stage == 'run' and run.test.kind != 'external':
                        run.status[2] = run_and_compare(run, ctx, log, state)
        except Exception as e:  # a problem of the test system itself must not stop other runs
            stage = next((i for i, st in enumerate(run.status) if st == NONE), 2)
            run.status[stage] = FAIL
            run.messages.append(f'internal error of ktest: {type(e).__name__}: {e}')
        if run.messages:
            log.write('\n[ktest]: ' + '\n[ktest]: '.join(run.messages) + '\n')
    run.seconds = time.time() - t0
    with ctx.print_lock:
        ctx.done += 1
        line = f'{ctx.done:>4}/{ctx.total} ' + status_line(run, ctx.color) + f'  ({run.seconds:.1f} s)'
        if (FAIL in run.status or run.status[0] == SKIP or ctx.verbose) and run.messages:
            shown = run.messages if ctx.verbose else run.messages[:1]
            line += ''.join('\n            ' + m for m in shown)
        print(line, flush=True)


def execute_job(runs, ctx):
    state = {'cleaned': set()}
    for r in runs:
        if ctx.cancelled:
            return
        if r.backend != state.get('backend'):  # other back-end means other build folder and other executable
            state = {'cleaned': state['cleaned'], 'backend': r.backend, 'cpu_done': state.get('cpu_done')}
        execute_run(r, ctx, state)


def run_all(runs, ctx, jobs):
    groups = {}
    for r in runs:
        groups.setdefault(r.test.folder, []).append(r)
    ctx.total = len(runs)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(execute_job, g, ctx) for g in groups.values()]
        try:
            for f in futures:
                f.result()
        except KeyboardInterrupt:
            ctx.cancelled = True   # running commands get Ctrl+C too; jobs do not start new runs
            print('\ninterrupted, waiting for running commands to stop ...', flush=True)
            raise


def is_failed(run):
    return FAIL in run.status


def write_summary(runs, ctx, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in runs:
            f.write(status_line(r, False) + '\n')
            for m in r.messages:
                f.write('    ' + m + '\n')


def save_failed(runs, path):
    data = [{'folder': r.test.folder, 'kind': r.test.kind, 'variant': r.variant['name'], 'megakernel': r.megakernel,
             'backend': r.backend, 'source': r.test.source, 'cmake_dir': r.test.cmake_dir}
            for r in runs if is_failed(r)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
