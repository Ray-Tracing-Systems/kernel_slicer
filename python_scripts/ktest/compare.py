"""Comparison of CPU and GPU outputs of a sample: 'zout_*' json files and BMP images.
Thresholds are the same as in the old test system (python_scripts/compare_json.py, compare_images.py)."""
import os
import struct

from . import jsonc

OK, WARN, FAIL = 'OK', 'WARN', 'FAIL'
_RANK = {OK: 0, WARN: 1, FAIL: 2}

FLOAT_WARN, FLOAT_FAIL = 1e-4, 1e-2  # absolute error of json values
MSE_WARN,   MSE_FAIL   = 1e-4, 1e-2  # mse of images, colors in [0,1]


def worst(statuses):
    res = OK
    for s in statuses:
        if _RANK[s] > _RANK[res]:
            res = s
    return res


def _float_status(err):
    if err > FLOAT_FAIL:
        return FAIL
    if err > FLOAT_WARN:
        return WARN
    return OK


def _compare_values(name, a, b, msgs):
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            msgs.append(f'"{name}": types differ')
            return FAIL
        err = abs(a - b)
        st = _float_status(err)
        if st != OK:
            msgs.append(f'"{name}": float error {err:g}')
        return st
    if isinstance(a, list) and isinstance(b, list) and a and isinstance(a[0], float):
        if len(a) != len(b):
            msgs.append(f'"{name}": array sizes differ ({len(a)} vs {len(b)})')
            return FAIL
        errs = [abs(x - y) for x, y in zip(a, b)]
        st = OK
        for err, kind in ((sum(errs) / len(errs), 'mean'), (max(errs), 'max')):
            s = _float_status(err)
            if s != OK:
                msgs.append(f'"{name}": {kind} float error {err:g}')
            st = worst([st, s])
        return st
    if a != b:
        msgs.append(f'"{name}": values are not equal')
        return FAIL
    return OK


def compare_json(path_a, path_b, msgs):
    try:
        ja, jb = jsonc.load(path_a), jsonc.load(path_b)
    except Exception as e:
        msgs.append(f'can not read json: {e}')
        return FAIL
    st = OK
    if ja.keys() != jb.keys():
        msgs.append(f'different sets of keys: {sorted(set(ja) ^ set(jb))}')
        st = FAIL
    for key in ja:
        if key in jb:
            st = worst([st, _compare_values(key, ja[key], jb[key], msgs)])
    return st


def read_bmp(path):
    """returns (width, height, rgb bytes) for uncompressed 24/32 bit BMP"""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] != b'BM':
        raise ValueError('not a BMP file')
    offset = struct.unpack_from('<I', data, 10)[0]
    w, h = struct.unpack_from('<ii', data, 18)
    bpp, compression = struct.unpack_from('<HI', data, 28)
    if bpp not in (24, 32) or compression not in (0, 3):
        raise ValueError(f'unsupported BMP format: {bpp} bpp, compression {compression}')
    h = abs(h)
    bytes_pp = bpp // 8
    stride = (w * bytes_pp + 3) & ~3
    rows = []
    for y in range(h):
        row = data[offset + y * stride: offset + y * stride + w * bytes_pp]
        if bytes_pp == 4:
            row = bytes(b for i, b in enumerate(row) if i % 4 != 3)
        rows.append(row)
    return w, h, b''.join(rows)


def compare_bmp(path_a, path_b, msgs):
    try:
        wa, ha, a = read_bmp(path_a)
        wb, hb, b = read_bmp(path_b)
    except Exception as e:
        msgs.append(f'can not read image: {e}')
        return FAIL
    if (wa, ha) != (wb, hb):
        msgs.append(f'image sizes differ: {wa}x{ha} vs {wb}x{hb}')
        return FAIL
    sq = sum((x - y) * (x - y) for x, y in zip(a, b))
    mse = sq / (255.0 * 255.0) / max(1, len(a))
    st = FAIL if mse > MSE_FAIL else (WARN if mse > MSE_WARN else OK)
    if st != OK:
        msgs.append(f'{os.path.basename(path_a)} vs {os.path.basename(path_b)}: mse = {mse:g}')
    return st


def compare_outputs(folder, cpu_files, gpu_tag, msgs):
    """cpu_files -- names of 'zout_*cpu*' files; the pair is the same name with 'cpu' replaced by gpu_tag"""
    statuses = []
    for name in sorted(cpu_files):
        other = name.replace('cpu', gpu_tag, 1)
        pa, pb = os.path.join(folder, name), os.path.join(folder, other)
        if not os.path.isfile(pb):
            msgs.append(f'{other} is missing')
            statuses.append(FAIL)
        elif name.endswith('.json'):
            statuses.append(compare_json(pa, pb, msgs))
        elif name.lower().endswith('.bmp'):
            statuses.append(compare_bmp(pa, pb, msgs))
        else:
            msgs.append(f'{name}: unsupported file type, not compared')
            statuses.append(WARN)
    if not statuses:
        msgs.append('no zout_* files to compare')
        statuses.append(WARN)
    return worst(statuses)
