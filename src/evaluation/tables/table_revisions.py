"""Compare reviewed LaTeX tables against a frozen manuscript baseline."""
from decimal import Decimal
from difflib import SequenceMatcher
import hashlib
import json
import re

BLUE = r'\textcolor{blue!60!black}'


def group_end(text, start):
    """Return the end of a balanced TeX brace group, respecting escapes."""
    if text[start] != '{':
        raise ValueError('Expected a brace group')
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '\\':
            i += 2
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError('Unbalanced TeX braces')


def strip_marks(text):
    """Remove revision wrappers only; retain boldface, shading, and other colors."""
    pattern = re.compile(r'\\update\s*(?=\{)|\\textcolor\s*\{blue(?:![^{}]*)?\}\s*(?=\{)')
    while match := pattern.search(text):
        start = match.end()
        end = group_end(text, start)
        text = text[:match.start()] + text[start + 1:end - 1] + text[end:]
    return text


def extract_table(text, label):
    marker = r'\label{' + label + '}'
    if text.count(marker) != 1:
        raise ValueError(f'Expected exactly one {marker}')
    pos = text.index(marker)
    starts = list(re.finditer(r'\\begin\{table\*?\}', text[:pos]))
    if not starts:
        raise ValueError(f'Missing table environment for {label}')
    start = starts[-1]
    end = start.group().replace('begin', 'end')
    return text[start.start():text.index(end, pos) + len(end)] + '\n'


def freeze_base(base_dir, main_tex, appendix_tex, tables):
    """Create a baseline once, from the original paper; never overwrite one."""
    if base_dir.exists() and any(base_dir.iterdir()):
        raise FileExistsError(f'Baseline already exists: {base_dir}; use a new --base-dir to create another')
    sources = {'main': main_tex.read_text(), 'appendix': appendix_tex.read_text()}
    blocks = {key: strip_marks(extract_table(sources['main' if key == 'main' else 'appendix'], label))
              for key, label in tables.items()}
    base_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key, block in blocks.items():
        (base_dir / (key + '.tex')).write_text(block)
        hashes[key + '.tex'] = hashlib.sha256(block.encode()).hexdigest()
    metadata = {'description': 'Original manuscript tables, before evaluation corrections',
                'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [main_tex, appendix_tex]}, 'tables': hashes}
    (base_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


def load_base(base_dir, tables):
    metadata = json.loads((base_dir / 'manifest.json').read_text())
    blocks = {}
    for key, label in tables.items():
        name = key + '.tex'
        data = (base_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata['tables'][name]:
            raise ValueError(f'Frozen baseline was modified: {name}')
        blocks[key] = data.decode()
        if blocks[key].count(r'\label{' + label + '}') != 1:
            raise ValueError(f'Incorrect baseline table label: {name}')
    return blocks


def cell_value(cell, significance=False):
    """Compare displayed values, ignoring boldface and cosmetic whitespace."""
    text = strip_marks(cell).strip()
    shade = re.fullmatch(r'\\shadeA\{([^{}]*)\}', text)
    if significance:
        if shade:
            p = shade.group(1).strip()
            if not p:
                return ('significance', 0)
            p = Decimal(p)
            return ('significance', 3 if p < Decimal('.001') else 2 if p < Decimal('.01') else 1 if p < Decimal('.05') else 0)
        if text in {'', '--'}:
            return ('significance', 0)
        if text in {'*', '**', '***'}:
            return ('significance', len(text))
        raise ValueError(f'Unrecognized significance cell: {cell}')
    for command in ['textbf', 'mathbf', 'mathrm']:
        text = re.sub(r'\\' + command + r'\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'[\s${}]', '', text)
    if text in {'', '--'}:
        return ('missing',)
    match = re.fullmatch(r'(<|>)?([+-]?\d+(?:\.\d+)?)(\*{0,3})', text)
    if not match:
        raise ValueError(f'Unrecognized numerical cell: {cell}')
    return (match.group(1), Decimal(match.group(2)), match.group(3))


def rows(text, key):
    """Identify data rows by model/strategy (or r/p), never by row position."""
    result = {}
    carry = None
    for index, line in enumerate(text.splitlines(keepends=True)):
        ending = re.search(r'\\\\(?:\s*\\midrule)?\s*$', line)
        if not ending:
            continue
        cells = line[:ending.start()].split('&')
        if key in {'main', 'filtered'}:
            if len(cells) != 15 or cells[2].strip() not in {'Base', 'CtE', 'Ora.'}:
                continue
            carry = cells[0].strip() or carry
            identity, prefix = (carry, cells[2].strip()), 3
        elif key == 'classification':
            if len(cells) != 8 or not re.fullmatch('[A-Z]', cells[0].strip()):
                continue
            identity, prefix = (cells[0].strip(),), 2
        elif key == 'correlation':
            if len(cells) != 10 or cells[0].strip() not in {'$r$', '$p$'}:
                continue
            identity, prefix = (cells[0].strip(),), 1
        elif key == 'significance':
            if len(cells) != 11:
                continue
            try:
                for cell in cells[1:]:
                    cell_value(cell, significance=True)
            except ValueError:
                continue
            identity, prefix = (re.sub(r'\s+', '', cells[0]),), 1
        else:
            if len(cells) != (7 if key == 'cfr' else 10) or cells[1].strip() not in {'Base', 'CtE'}:
                continue
            carry = re.sub(r'\s+', '', cells[0]) or carry
            identity, prefix = (carry, cells[1].strip()), 2
        if identity in result:
            raise ValueError(f'Duplicate row in {key}: {identity}')
        result[identity] = (index, cells, prefix, line[ending.start():])
    if not result:
        raise ValueError(f'No recognized data rows in {key}')
    return result


def atoms(text):
    """Token spans with complete TeX commands/groups and math kept intact."""
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        if text[i] == '$':
            end = text.find('$', i + 1)
            if end < 0:
                raise ValueError('Unbalanced caption math')
            i = end + 1
        elif text[i] == '\\':
            match = re.match(r'\\(?:[A-Za-z]+\*?|.)', text[i:])
            i += len(match.group())
            while i < len(text) and text[i] == '{':
                i = group_end(text, i)
        elif text[i] == '{':
            i = group_end(text, i)
        else:
            match = re.match(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|.", text[i:])
            i += len(match.group())
        result.append((text[start:i], start, i))
    return result


def caption_span(text):
    match = re.search(r'\\caption\s*(?=\{)', text)
    if not match:
        raise ValueError('Missing caption')
    a = match.end()
    return a + 1, group_end(text, a) - 1


def mark_caption(base, current):
    a, b = caption_span(base)
    c, d = caption_span(current)
    old, new = atoms(base[a:b]), atoms(current[c:d])
    edits = []
    for tag, _, _, j, k in SequenceMatcher(None, [x[0] for x in old], [x[0] for x in new], autojunk=False).get_opcodes():
        if tag in {'insert', 'replace'}:
            for value, start, end in new[j:k]:
                edits.append((c + start, c + end, BLUE + '{' + value + '}'))
    for start, end, value in reversed(edits):
        current = current[:start] + value + current[end:]
    return current


def compare_table(base, current, key):
    clean = strip_marks(current)
    old_rows, new_rows = rows(base, key), rows(clean, key)
    if old_rows.keys() != new_rows.keys():
        raise ValueError(f'Row identities changed in {key}; review added/removed/renamed rows before exporting')
    def values(data):
        resolved, previous = {}, {}
        for identity, (_, cells, prefix, _) in data.items():
            resolved[identity] = {}
            for col in range(prefix, len(cells)):
                if cells[col].strip() == r'\ditto':
                    if col not in previous:
                        raise ValueError('Ditto cell has no preceding value')
                    value = previous[col]
                else:
                    value = cell_value(cells[col], key == 'significance')
                resolved[identity][col] = previous[col] = value
        return resolved
    old_values, new_values = values(old_rows), values(new_rows)
    lines = clean.splitlines(keepends=True)
    changes = []
    for identity, (line_index, cells, prefix, ending) in new_rows.items():
        before = old_rows[identity][1]
        for col in range(prefix, len(cells)):
            old_value = old_values[identity][col]
            new_value = new_values[identity][col]
            if old_value != new_value:
                value = cells[col].strip()
                changes.append({'row': list(identity), 'column': col - prefix + 1,
                                'base': before[col].strip(), 'current': value})
                cells[col] = ' ' + BLUE + '{' + value + '} '
        lines[line_index] = '&'.join(cells) + ending
    blue = mark_caption(base, ''.join(lines))
    if strip_marks(blue) != clean:
        # Cell spacing is cosmetic, but all other source content must survive.
        if re.sub(r'\s+', '', strip_marks(blue)) != re.sub(r'\s+', '', clean):
            raise AssertionError(f'Coloring changed table content: {key}')
    return clean, blue, changes


def export_versions(current, base_dir, outdir, selected, tables):
    base = load_base(base_dir, tables)
    generated = {key: compare_table(base[key], current[key], key) for key in selected}
    if outdir.resolve() == base_dir.resolve():
        raise ValueError('Output directory must differ from the frozen baseline')
    for variant in ['clean', 'blue']:
        destination = outdir / variant
        if destination.resolve() == base_dir.resolve():
            raise ValueError('Output would overwrite the frozen baseline')
    report = {'base_dir': str(base_dir), 'tables': {}}
    for key, (clean, blue, changes) in generated.items():
        for variant, text in [('clean', clean), ('blue', blue)]:
            destination = outdir / variant / (key + '.tex')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
        a, b = caption_span(base[key])
        c, d = caption_span(clean)
        report['tables'][key] = {
            'base_sha256': hashlib.sha256(base[key].encode()).hexdigest(),
            'current_sha256': hashlib.sha256(clean.encode()).hexdigest(),
            'caption_changed': [t[0] for t in atoms(base[key][a:b])] != [t[0] for t in atoms(clean[c:d])],
            'changed_cells': len(changes), 'changes': changes}
        print(f'{key}: {len(changes)} changed cells -> clean/{key}.tex, blue/{key}.tex')
    (outdir / 'changes.json').write_text(json.dumps(report, indent=2) + '\n')
