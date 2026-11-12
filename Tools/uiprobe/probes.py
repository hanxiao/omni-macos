"""probes.py add|remove: a temporary UIProbe.count("body.X") at the top of every view, modifier,
scene and commands body under App/. Every inserted line ends with the marker and is removed by it."""
import re, sys, glob
MARK = '// RENDER-PROBE'
body = re.compile(r'^(\s*)(?:@\w+\s+)*(?:public |private |fileprivate |nonisolated )*(?:var body: some (?:View|Scene|Commands)|func body\(content: Content\) -> some View) \{\s*$')
typ = re.compile(r'^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public |private |fileprivate |final )*(?:struct|class|extension|enum) (\w+)')
n = 0
for p in sorted(glob.glob('App/**/*.swift', recursive=True)):
    lines = open(p).read().split('\n')
    if sys.argv[1] == 'remove':
        kept = [l for l in lines if not l.rstrip().endswith(MARK)]
        n += len(lines) - len(kept)
        if len(kept) != len(lines): open(p, 'w').write('\n'.join(kept))
        continue
    out, name, seen = [], '?', {}
    for i, l in enumerate(lines):
        m = typ.match(l)
        if m: name = m.group(1)
        out.append(l)
        m = body.match(l)
        if m and 'UIProbe.count' not in (lines[i + 1] if i + 1 < len(lines) else ''):
            seen[name] = seen.get(name, 0) + 1
            tag = name if seen[name] == 1 else f'{name}#{seen[name]}'
            out.append(f'{m.group(1)}    let _ = UIProbe.count("body.{tag}") {MARK}')
            n += 1
    if out != lines: open(p, 'w').write('\n'.join(out))
print(sys.argv[1], n)
