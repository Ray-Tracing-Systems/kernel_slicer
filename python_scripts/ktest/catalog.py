"""Discovery of tests (apps/tests/NNN_*), test applications (apps/NN_*) and external projects.

Every test folder contains 'ktest.json':
{
    "variants": [
        {"name": "slang", "backend": "slang", "kmake": "kmake.json", "megakernel": true},
        {"name": "glsl_hw", "backend": "glsl", "kmake": "kmake_hw.json", "options": {"-enableSubgroup": 0}}
    ],
    "requires": ["../resources/HydraCore"]   // optional: external resources needed to run the sample
}

"kmake"      -- kernel_slicer config (relative to the test folder);
"backend"    -- slang | glsl | ispc | cuda, passed to kslicer as '-shaderCC';
"options"    -- optional kslicer options added on the command line (override options of the config);
"megakernel" -- if true, the variant is tested twice: with '-megakernel 0' and '-megakernel 1';
"skip"       -- optional reason why the variant is known not to work yet: it is reported as SKIP
                (run it anyway with --no-skip).
"""
import os
import re

from . import jsonc

BACKENDS = ['slang', 'glsl', 'ispc', 'cuda']

# known external resources for test applications: folder relative to 'apps' -> (git url, branch)
RESOURCES = {
    'resources/msu-graphics-group/scenes': ('https://github.com/msu-graphics-group/scenes.git', 'main'),
    'resources/HydraCore':                 ('https://github.com/Ray-Tracing-Systems/HydraCore.git', 'master'),
}


class Run:
    """One kslicer call + build + run for a single variant of a test"""
    def __init__(self, test, variant, megakernel):
        self.test       = test
        self.variant    = variant
        self.megakernel = megakernel  # None, 0 or 1
        self.name       = variant['name'] + ('' if megakernel is None else f'/mk{megakernel}')
        self.backend    = variant['backend']
        self.kmake      = os.path.normpath(os.path.join(test.folder, variant['kmake']))
        self.options    = variant.get('options', {})
        self.skip       = variant.get('skip')  # reason of a known problem or None
        self.status     = ['--', '--', '--']  # GEN | BUILD | CMP
        self.messages   = []
        self.log_file   = None
        self.seconds    = 0.0

    @property
    def full_name(self):
        return f'{self.test.name}/{self.name}'

    def kslicer_args(self):
        """kslicer arguments except the program itself; paths are absolute"""
        args = [self.kmake, '-shaderCC', self.backend]
        for key, val in self.options.items():
            for v in (val if isinstance(val, list) else [val]):
                args += [key, str(v)]
        if self.megakernel is not None:
            args += ['-megakernel', str(self.megakernel)]
        return args

    def option(self, key, default=None):
        """value of kslicer option taking into account both variant options and the config"""
        if key in self.options:
            return str(self.options[key])
        opts = self.test.kmake_options(self.kmake)
        if key in opts:
            return str(opts[key])
        return default

    def shader_folder(self):
        suffix = self.option('-suffix', '_Generated')  # the same as 'ShaderFolder()' in kslicer.h
        return os.path.join(self.test.folder, 'shaders' + suffix.lower())


class Test:
    def __init__(self, folder, kind, desc, root):
        self.folder   = folder
        self.kind     = kind                       # 'test' | 'app' | 'external'
        self.dirname  = os.path.basename(folder)
        m = re.match(r'(\d+)_', self.dirname)
        self.number   = int(m.group(1)) if m else None
        self.name     = (f'{kind}_{self.dirname}' if kind == 'external' else
                         ('test_' if kind == 'test' else 'app_') + self.dirname)
        self.desc     = desc
        self.root     = root
        self.requires = [os.path.normpath(os.path.join(folder, r)) for r in desc.get('requires', [])]
        self.cmake_dir = folder
        self.source    = folder                     # what external project was created from
        self._kmake_cache = {}

    def kmake_options(self, kmake_path):
        if kmake_path not in self._kmake_cache:
            try:
                self._kmake_cache[kmake_path] = jsonc.load(kmake_path).get('options', {}) or {}
            except Exception:
                self._kmake_cache[kmake_path] = {}
        return self._kmake_cache[kmake_path]

    def runs(self, backends, variant_filter=None, megakernel=None):
        res = []
        for v in self.desc.get('variants', []):
            if v['backend'] not in backends:
                continue
            if variant_filter and not any(re.search(f, v['name']) for f in variant_filter):
                continue
            modes = [0, 1] if v.get('megakernel') else [None]
            for mk in modes:
                if megakernel is not None and mk is not None and mk != megakernel:
                    continue
                res.append(Run(self, v, mk))
        return res

    def missing_resources(self):
        return [r for r in self.requires if not os.path.isdir(r)]


def _load(folder, kind, root):
    path = os.path.join(folder, 'ktest.json')
    if not os.path.isfile(path):
        return None
    return Test(folder, kind, jsonc.load(path), root)


def discover(root, kinds):
    """kinds -- subset of {'test', 'app'}"""
    tests = []
    apps_dir = os.path.join(root, 'apps')
    if 'test' in kinds:
        tdir = os.path.join(apps_dir, 'tests')
        for d in sorted(os.listdir(tdir)):
            if re.match(r'\d{3}_', d):
                t = _load(os.path.join(tdir, d), 'test', root)
                if t:
                    tests.append(t)
    if 'app' in kinds:
        for d in sorted(os.listdir(apps_dir)):
            if re.match(r'\d{2}_', d):
                t = _load(os.path.join(apps_dir, d), 'app', root)
                if t:
                    tests.append(t)
    return tests


def external(root, kmake, cmake_dir, backends):
    """External project: folder with ktest.json, or a single kmake.json (+ folder with CMakeLists.txt)"""
    kmake = os.path.abspath(kmake)
    if os.path.isdir(kmake):
        folder = kmake
        desc = jsonc.load(os.path.join(folder, 'ktest.json'))
    elif os.path.basename(kmake) == 'ktest.json':
        folder = os.path.dirname(kmake)
        desc = jsonc.load(kmake)
    else:
        folder = os.path.dirname(kmake)
        opts = jsonc.load(kmake).get('options', {}) or {}
        cfg_backend = str(opts.get('-shaderCC', 'slang')).lower()
        bes = backends if backends else [cfg_backend]
        desc = {'variants': [{'name': be, 'backend': be, 'kmake': os.path.basename(kmake)} for be in bes]}
    t = Test(folder, 'external', desc, root)
    t.cmake_dir = os.path.abspath(cmake_dir) if cmake_dir else folder
    t.source    = kmake
    return t


def parse_filter(text):
    """'23-30,45,app_05' -> list of (kind or None, lo, hi) and list of name substrings"""
    ranges, names = [], []
    for item in re.split(r'[,\s]+', text.strip()):
        if not item:
            continue
        m = re.fullmatch(r'(?:(test|app)_?)?(\d+)(?:-(\d+))?', item)
        if m:
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            ranges.append((m.group(1), lo, hi))
        else:
            names.append(item)
    return ranges, names


def apply_filter(tests, text):
    if not text:
        return tests
    ranges, names = parse_filter(text)
    res = []
    for t in tests:
        ok = any((kind is None or kind == t.kind) and t.number is not None and lo <= t.number <= hi
                 for kind, lo, hi in ranges)
        ok = ok or any(n in t.name for n in names)
        if ok:
            res.append(t)
    return res
