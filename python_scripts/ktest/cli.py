"""Command line and interactive menu of the test system. Entry point: ktest.py in the repository root."""
import os
import sys
import glob
import json
import time
import shutil
import argparse
import subprocess

from . import catalog, runner, launch
from .catalog import BACKENDS, RESOURCES

IS_WINDOWS = os.name == 'nt'

EPILOG = """examples:
  python3 ktest.py                                  interactive menu
  python3 ktest.py -b slang                         all tests (apps/tests) with Slang
  python3 ktest.py -b slang,glsl -s all -j 8        tests and test applications, two back-ends
  python3 ktest.py -b slang --only 23-26,45         only tests 023..026 and 045
  python3 ktest.py -s apps --only 5 --stage build   generate and build app 05, do not run
  python3 ktest.py --rerun-failed --write-launch    rerun failed runs, write .vscode/launch_failed.json
  python3 ktest.py --external ~/proj/kmake.json --cmake-dir ~/proj   generate and build external project
"""


def find_kslicer(root):
    exe = 'kslicer.exe' if IS_WINDOWS else 'kslicer'
    for d in ('cmake-build-release', 'cmake-build-release/Release', 'build', 'build/Release',
              'cmake-build-debug', 'cmake-build-debug/Debug'):
        p = os.path.join(root, d, exe)
        if os.path.isfile(p):
            return p
    return None


def check_kslicer_age(root, kslicer):
    newest, newest_file = 0, None
    for f in glob.glob(os.path.join(root, '*.cpp')) + glob.glob(os.path.join(root, '*.h')):
        m = os.path.getmtime(f)
        if m > newest:
            newest, newest_file = m, f
    if newest_file and newest > os.path.getmtime(kslicer):
        print(f'warning: {os.path.relpath(kslicer, root)} is older than {os.path.basename(newest_file)}, '
              f'probably kslicer should be rebuilt')


def parse_args(argv):
    p = argparse.ArgumentParser(prog='ktest.py', description='kernel_slicer test system: generation, build, '
                                'run on CPU and GPU and comparison of results. Without options which select tests '
                                '(-b, -s, --only, --variant, --external, --rerun-failed, --list) shows a menu.',
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-b', '--backend', help='comma separated list of: ' + ', '.join(BACKENDS) + ', all (default: slang; '
                   'for --external: back-end of the config)')
    p.add_argument('-s', '--set', default='tests', choices=['tests', 'apps', 'all'],
                   help='tests (apps/tests), apps (test applications apps/NN_*) or all (default: tests)')
    p.add_argument('--only', default='', help="numbers, ranges or names of tests: '23-30,45', 'app_05', 'hash_map'")
    p.add_argument('--variant', action='append', help='regular expression for variant names (may be repeated)')
    p.add_argument('--megakernel', type=int, choices=[0, 1], help='for variants with megakernel run only this mode')
    p.add_argument('--stage', default='run', choices=['gen', 'build', 'run'], help='last stage to execute (default: run)')
    p.add_argument('-j', '--jobs', type=int, default=0, help='number of test folders processed in parallel')
    p.add_argument('--gpu-jobs', type=int, default=1, help='number of samples running on GPU at the same time (default: 1)')
    p.add_argument('--build-jobs', type=int, default=0, help='parallel jobs for every cmake build and shader script')
    p.add_argument('--kslicer', help='path to kslicer executable (default: cmake-build-release/kslicer)')
    p.add_argument('--gpu-id', type=int, default=0, help='GPU id passed to samples')
    p.add_argument('--build-type', default='Release', help='CMAKE_BUILD_TYPE of samples (default: Release)')
    p.add_argument('--generator', help='cmake generator for samples, e.g. Ninja')
    p.add_argument('--cmake-args', default='', help="additional cmake arguments, e.g. '-DCMAKE_CXX_COMPILER=clang++'")
    p.add_argument('--clean', action='store_true', help='remove build folders of samples before build')
    p.add_argument('--timeout', type=int, default=1200, help='timeout of every command in seconds (default: 1200)')
    p.add_argument('--external', action='append', default=[], help='external project: path to kmake.json, ktest.json or a '
                   'folder with ktest.json; only generation and build are executed (may be repeated)')
    p.add_argument('--cmake-dir', action='append', default=[], help='folder with CMakeLists.txt of corresponding '
                   '--external project (default: folder of the config)')
    p.add_argument('--rerun-failed', action='store_true', help='run only runs which failed last time')
    p.add_argument('--write-launch', nargs='?', const='.vscode/launch_failed.json', metavar='PATH',
                   help='write VS Code debug configurations for failed runs (default: .vscode/launch_failed.json)')
    p.add_argument('--download', action='store_true', help='clone missing resources of test applications (git)')
    p.add_argument('--no-skip', action='store_true', help='also run variants marked with "skip" in ktest.json')
    p.add_argument('--list', action='store_true', help='only print selected runs')
    p.add_argument('--no-color', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true', help='print all messages of every run')
    return p.parse_args(argv)


def ask(prompt, default):
    try:
        ans = input(f'{prompt} [{default}]: ').strip()
    except EOFError:
        ans = ''
    return ans or str(default)


def ask_numbers(prompt, default, maxval):
    while True:
        ans = ask(prompt, default).replace(',', ' ').split()
        try:
            nums = [int(a) for a in ans]
            if nums and all(1 <= n <= maxval for n in nums):
                return nums
        except ValueError:
            pass
        print(f'  please enter numbers from 1 to {maxval}')


def interactive(args, root):
    """fills args from a simple menu; returns the equivalent command line"""
    tools = {be: True for be in BACKENDS}
    tools['ispc'] = bool(shutil.which('ispc'))
    tools['cuda'] = bool(shutil.which('nvcc'))
    n_tests = len(catalog.discover(root, {'test'}))
    n_apps = len(catalog.discover(root, {'app'}))
    print('kernel_slicer tests. Press Enter to accept the default value in [brackets].\n')
    print('Back-ends:')
    for i, be in enumerate(BACKENDS, 1):
        print(f'  {i}) {be}' + ('' if tools[be] else f'   (compiler is not found, only generation)'))
    nums = ask_numbers('Back-ends, several numbers are allowed', 1, len(BACKENDS))
    args.backend = ','.join(BACKENDS[n - 1] for n in nums)
    print('\nWhat to test:')
    print(f'  1) tests               apps/tests/NNN_*  ({n_tests})')
    print(f'  2) test applications   apps/NN_*         ({n_apps})')
    print('  3) both')
    print('  4) external project    path to kmake.json, generation and build only')
    what = ask_numbers('Choice', 1, 4)[0]
    cmd = ['-b', args.backend]
    if what == 4:
        path = ''
        while not os.path.exists(path):
            path = os.path.expanduser(ask('Path to kmake.json, ktest.json or folder', ''))
            if not os.path.exists(path):
                print(f'  "{path}" does not exist')
        args.external = [path]
        cmake_dir = ask('Folder with CMakeLists.txt (empty: folder of the config)', '-')
        args.cmake_dir = [os.path.expanduser(cmake_dir)] if cmake_dir != '-' else []
        cmd += ['--external', args.external[0]] + (['--cmake-dir', args.cmake_dir[0]] if args.cmake_dir else [])
    else:
        args.set = ['tests', 'apps', 'all'][what - 1]
        cmd += ['-s', args.set]
        only = ask("Only these numbers/ranges/names, e.g. '23-30,45'", 'all')
        if only != 'all':
            args.only = only
            cmd += ['--only', only]
        print('\nWhat to do:')
        print('  1) generate + build + run on CPU and GPU + compare')
        print('  2) generate + build')
        print('  3) generate only')
        stage = ask_numbers('Choice', 1, 3)[0]
        args.stage = ['run', 'build', 'gen'][stage - 1]
        if args.stage != 'run':
            cmd += ['--stage', args.stage]
    jobs = ask_numbers('\nParallel jobs', default_jobs(), 256)[0]
    if jobs != default_jobs():
        cmd += ['-j', str(jobs)]
    args.jobs = jobs
    return cmd


def default_jobs():
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def download_resources(tests, root):
    for t in tests:
        for r in t.missing_resources():
            key = os.path.relpath(r, os.path.join(root, 'apps')).replace(os.sep, '/')
            if key not in RESOURCES:
                print(f'unknown resource {r}, please download it manually')
                continue
            url, branch = RESOURCES[key]
            print(f'cloning {url} -> {r}')
            subprocess.run(['git', 'clone', '--depth', '1', '-b', branch, url, r])


def select_runs(args, root):
    text = args.backend or 'slang'
    backends = BACKENDS if text == 'all' else [b.strip().lower() for b in text.split(',')]
    for b in backends:
        if b not in BACKENDS:
            sys.exit(f'unknown back-end "{b}", known: {", ".join(BACKENDS)}')
    if args.external:
        tests = [catalog.external(root, e, args.cmake_dir[i] if i < len(args.cmake_dir) else None,
                                  backends if args.backend else None)
                 for i, e in enumerate(args.external)]
        backends = BACKENDS
    else:
        kinds = {'tests': {'test'}, 'apps': {'app'}, 'all': {'test', 'app'}}[args.set]
        tests = catalog.apply_filter(catalog.discover(root, kinds), args.only)
    runs = []
    for t in tests:
        runs += t.runs(backends, args.variant, args.megakernel)
    return tests, runs


def select_failed(root, log_root):
    path = os.path.join(log_root, 'last_failed.json')
    if not os.path.isfile(path):
        sys.exit(f'{path} is not found: nothing to rerun')
    with open(path, encoding='utf-8') as f:
        failed = json.load(f)
    tests, runs = {}, []
    for item in failed:
        folder = item['folder']
        if item['kind'] != 'external' and not os.path.isfile(os.path.join(folder, 'ktest.json')):
            print(f'warning: {folder}/ktest.json is not found, skipped')
            continue
        if folder not in tests:
            if item['kind'] == 'external':
                tests[folder] = catalog.external(root, item['source'], item['cmake_dir'], [item['backend']])
            else:
                tests[folder] = catalog.Test(folder, item['kind'], catalog.jsonc.load(os.path.join(folder, 'ktest.json')), root)
        for r in tests[folder].runs(BACKENDS):
            if r.variant['name'] == item['variant'] and r.megakernel == item['megakernel']:
                runs.append(r)
    return list(tests.values()), runs


def main(argv, root):
    if IS_WINDOWS:
        os.system('')  # enables ANSI colors in Windows console
    interactive_cmd = None
    args = parse_args(argv)
    selection = {'-b', '--backend', '-s', '--set', '--only', '--variant', '--megakernel', '--external', '--rerun-failed', '--list'}
    if sys.stdin.isatty() and not any(a.split('=')[0] in selection for a in argv):
        interactive_cmd = interactive(args, root)  # options like --kslicer, -j or --no-color do not disable the menu

    kslicer = os.path.abspath(args.kslicer) if args.kslicer else find_kslicer(root)
    if not kslicer or not os.path.isfile(kslicer):
        sys.exit('kslicer executable is not found. Build it first (see README.md, "Build") or pass --kslicer <path>.')
    check_kslicer_age(root, kslicer)

    log_root = os.path.join(root, 'ktest_logs')
    if args.rerun_failed:
        tests, runs = select_failed(root, log_root)
    else:
        tests, runs = select_runs(args, root)
    if not runs:
        sys.exit('nothing to run: check --backend, --set and --only')

    if args.list:
        for r in runs:
            print(r.full_name, ' '.join(os.path.relpath(a, root) if os.path.isabs(a) else a for a in r.kslicer_args()) +
                  (f'   [skip: {r.skip}]' if r.skip else ''))
        return 0

    if args.download:
        download_resources(tests, root)
    elif args.stage == 'run':
        missing = sorted({os.path.relpath(m, root) for t in tests for m in t.missing_resources()})
        if missing:
            print('missing resources (runs of these apps will be skipped, use --download to clone them):')
            for m in missing:
                print('  ' + m)

    jobs = args.jobs or default_jobs()
    jobs = min(jobs, len({r.test.folder for r in runs}))
    args.build_jobs = args.build_jobs or max(2, (os.cpu_count() or 2) // jobs)
    log_dir = os.path.join(log_root, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    ctx = runner.Context(root, kslicer, log_dir, args)

    print(f'kslicer : {os.path.relpath(kslicer, root)}')
    print(f'runs    : {len(runs)} in {len({r.test.folder for r in runs})} folders, {jobs} parallel jobs')
    print(f'logs    : {os.path.relpath(log_dir, root)}')
    print('status  : [generation|build|comparison with CPU]\n')
    t0 = time.time()
    try:
        runner.run_all(runs, ctx, jobs)
    except KeyboardInterrupt:
        runs = [r for r in runs if r.status[0] != runner.NONE or r.messages]  # report only what was executed

    failed = [r for r in runs if runner.is_failed(r)]
    runner.write_summary(runs, ctx, os.path.join(log_dir, 'summary.txt'))
    runner.save_failed(runs, os.path.join(log_root, 'last_failed.json'))
    counts = {}
    for r in runs:
        key = 'FAIL' if runner.is_failed(r) else ('WARN' if 'WARN' in r.status else
                                                 ('SKIP' if 'SKIP' in r.status else 'OK'))
        counts[key] = counts.get(key, 0) + 1
    print(f'\nfinished in {time.time() - t0:.0f} s: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
    if failed:
        print('\nfailed:')
        for r in failed:
            print(f'  {runner.status_line(r, ctx.color)}  {os.path.relpath(r.log_file, root)}')
            for m in r.messages[:3]:
                print(f'      {m}')
        launch.write_launch(failed, root, os.path.join(log_dir, 'launch_failed.json'))
        target = args.write_launch
        if target is None and interactive_cmd is not None:
            if ask(f'\nWrite .vscode/launch_failed.json with {len(failed)} failed runs? (y/n)', 'n').lower().startswith('y'):
                target = '.vscode/launch_failed.json'
        if target:
            target = os.path.join(root, target) if not os.path.isabs(target) else target
            launch.write_launch(failed, root, target)
            print(f'debug configurations of failed runs: {os.path.relpath(target, root)}')
        print('to rerun only failed runs: python3 ktest.py --rerun-failed')
    if interactive_cmd is not None:
        print('\nthe same without menu: python3 ktest.py ' + ' '.join(interactive_cmd + argv))
    return 1 if failed else 0
