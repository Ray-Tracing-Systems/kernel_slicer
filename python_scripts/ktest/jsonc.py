"""Reading JSON with comments and trailing commas (VS Code launch.json, kmake.json) using only the standard library."""
import json


def strip_jsonc(text):
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return _remove_trailing_commas(''.join(out))


def _remove_trailing_commas(text):
    out = []
    in_str = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c == ',':
            j = i + 1
            while j < n and text[j] in ' \t\r\n':
                j += 1
            if j < n and text[j] in ']}':
                pass  # drop trailing comma
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def loads(text):
    return json.loads(strip_jsonc(text))


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return loads(f.read())
