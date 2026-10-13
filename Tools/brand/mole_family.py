# The Omni mole glyph family: one drawing, one gag per screen. Output: <variant>.svg
import sys
def star(cx, cy, r):   # four-point sparkle, concave sides
    return (f'M{cx},{cy-r} Q{cx},{cy} {cx+r},{cy} Q{cx},{cy} {cx},{cy+r} '
            f'Q{cx},{cy} {cx-r},{cy} Q{cx},{cy} {cx},{cy-r} Z')
def svg(v):
    eyes = '<circle cx="40" cy="39" r="3.4" fill="black"/><circle cx="60" cy="39" r="3.4" fill="black"/>'
    extra_mask, extra_fill, paws = "", "", [(33, 71), (67, 71)]
    if v == "search":
        eyes = f'<path d="{star(40,39,6.2)}" fill="black"/><path d="{star(60,39,6.2)}" fill="black"/>'
        extra_fill = f'<path d="{star(84,20,6)}"/><path d="{star(92,32,3.4)}"/>'
    elif v == "ocr":
        eyes = ('<circle cx="40" cy="39" r="2.6" fill="black"/><circle cx="60" cy="39" r="2.6" fill="black"/>'
                '<circle cx="40" cy="39" r="8" fill="none" stroke="black" stroke-width="2.4"/>'
                '<circle cx="60" cy="39" r="8" fill="none" stroke="black" stroke-width="2.4"/>'
                '<path d="M47.6,37.5 Q50,35.5 52.4,37.5" fill="none" stroke="black" stroke-width="2.2" stroke-linecap="round"/>'
                '<path d="M32,37.5 L26,35.5 M68,37.5 L74,35.5" fill="none" stroke="black" stroke-width="2.2" stroke-linecap="round"/>')
    elif v == "sleep":
        eyes = ('<path d="M36.2,39.5 Q40,43 43.8,39.5" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M56.2,39.5 Q60,43 63.8,39.5" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
        extra_fill = ('<path d="M80,14 L89,14 L80,23 L89,23" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
                      '<path d="M89.5,28 L95,28 L89.5,33.5 L95,33.5" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    elif v == "serious":
        # focused, determined: flat-topped eyes under brows angled down toward the nose
        eyes = ('<path d="M36.6,39 L43.4,39 A3.4,3.4 0 0 1 36.6,39 Z" fill="black"/>'
                '<path d="M56.6,39 L63.4,39 A3.4,3.4 0 0 1 56.6,39 Z" fill="black"/>'
                '<path d="M34.5,31.5 L45,35.5 M65.5,31.5 L55,35.5" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
    elif v == "empty":
        # x x eyes: nothing here
        eyes = ('<path d="M36.4,35.4 L43.6,42.6 M43.6,35.4 L36.4,42.6" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M56.4,35.4 L63.6,42.6 M63.6,35.4 L56.4,42.6" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
    elif v == "curious":
        # one brow up, one eye narrowed, a question in the air: nothing to show yet
        eyes = ('<path d="M36.6,39.6 L43.4,39.6 A3.4,3.4 0 0 1 36.6,39.6 Z" fill="black"/>'
                '<circle cx="60" cy="39" r="3.6" fill="black"/>'
                '<path d="M34.6,32.4 L45,35.2" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M54.6,31 Q59.5,25.2 65,29.6" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
        extra_fill = ('<path d="M81,17.5 Q81,11.5 87,11.5 Q93,11.5 93,17 Q93,21 87.5,23.5 L87.5,26.5" '
                      'fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
                      '<circle cx="87.5" cy="32" r="1.8"/>')
    elif v == "sad":
        # worried brows (inner ends up), a tear down each cheek, one more falling: no matches
        eyes = ('<circle cx="40" cy="40" r="3.2" fill="black"/><circle cx="60" cy="40" r="3.2" fill="black"/>'
                '<path d="M34.6,33.6 L44.6,30.2 M65.4,33.6 L55.4,30.2" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M37,44.6 Q33.2,50.6 37,53.2 Q40.8,50.6 37,44.6 Z" fill="black"/>'
                '<path d="M63,44.6 Q59.2,50.6 63,53.2 Q66.8,50.6 63,44.6 Z" fill="black"/>')
        extra_fill = '<path d="M82,47 Q77.4,54.4 82,58 Q86.6,54.4 82,47 Z"/>'
    elif v == "welcome":
        eyes = ('<path d="M36.2,41 Q40,36.5 43.8,41" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M56.2,41 Q60,36.5 63.8,41" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
        paws = [(67, 71)]
        # the raised arm grows out of the body's side; the gap runs only between arm and head
        extra_mask = '<path d="M22,62 Q12,60 13,45" fill="none" stroke="black" stroke-width="13" stroke-linecap="round"/>'
        extra_fill = ('<path d="M27,63 Q13,61 14,46" fill="none" stroke="black" stroke-width="8" stroke-linecap="round"/>'
                      '<ellipse cx="14.2" cy="42" rx="6" ry="7.4" transform="rotate(-8 14.2 42)"/>'
                      '<path d="M5.5,31 Q4,28 5.5,25 M10,28 Q9.5,25 11.5,22.5" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round"/>')
    pawcut = "".join(f'<ellipse cx="{x}" cy="{y}" rx="10" ry="7.2" fill="black"/>' for x, y in paws)
    pawfill = "".join(f'<ellipse cx="{x}" cy="{y}" rx="7.6" ry="5"/>' for x, y in paws)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
 <defs>
  <path id="rim" d="M6,88 C6,76 26,70 50,70 C74,70 94,76 94,88"/>
  <mask id="head"><rect width="100" height="100" fill="white"/>
   {eyes}
   <ellipse cx="50" cy="49.5" rx="6.8" ry="4.9" fill="black"/>
   <rect x="45.8" y="56" width="8.4" height="6.4" rx="1.6" fill="black"/><rect x="49.35" y="55" width="1.3" height="9" fill="white"/>
   <use href="#rim" fill="none" stroke="black" stroke-width="12"/>
   {pawcut}{extra_mask}
  </mask>
  <mask id="mound"><rect width="100" height="100" fill="white"/>{pawcut}</mask>
 </defs>
 <g mask="url(#head)">
  <circle cx="31" cy="25.5" r="4"/><circle cx="69" cy="25.5" r="4"/>
  <path d="M25,90 L25,45 C25,27 36,18 50,18 C64,18 75,27 75,45 L75,90 Z"/>
 </g>
 <path mask="url(#mound)" d="M9,89 C9,79 27,75 50,75 C73,75 91,79 91,89 C91,92 89,94 86,94 L14,94 C11,94 9,92 9,89 Z"/>
 {pawfill}{extra_fill}
</svg>'''
for v in ["base", "search", "ocr", "sleep", "serious", "empty", "curious", "sad"]:
    open(f"fam-{v}.svg", "w").write(svg(v))

# Two moles cheek to cheek: Linked Macs' "Linked with ..." sheet. The family's mole at the family's
# size on a 140x100 canvas, so it reads as the same drawing: happy eyes, leaning together, one mound,
# a heart. No arms: two of them read as too many hands at 68 pt. Monochrome by gaps: the front mole
# is cut out of the one behind with the family's 2.5-unit outline.
def hug():
    head = "M25,90 L25,45 C25,27 36,18 50,18 C64,18 75,27 75,45 L75,90 Z"
    ears = '<circle cx="31" cy="25.5" r="4"/><circle cx="69" cy="25.5" r="4"/>'
    face = ('<path d="M36.2,41 Q40,36.5 43.8,41" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
            '<path d="M56.2,41 Q60,36.5 63.8,41" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
            '<ellipse cx="50" cy="49.5" rx="6.8" ry="4.9" fill="black"/>'
            '<rect x="45.8" y="56" width="8.4" height="6.4" rx="1.6" fill="black"/><rect x="49.35" y="55" width="1.3" height="9" fill="white"/>')
    A, B = "translate(-8,0) rotate(7 50 90)", "translate(48,0) rotate(-7 50 90)"
    rim = "M4,88 C4,76 36,70 70,70 C104,70 136,76 136,88"
    paws = [(28, 72), (112, 72)]
    pawcut = "".join(f'<ellipse cx="{x}" cy="{y}" rx="10" ry="7.2" fill="black"/>' for x, y in paws)
    pawfill = "".join(f'<ellipse cx="{x}" cy="{y}" rx="7.6" ry="5"/>' for x, y in paws)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 100">
 <defs>
  <g id="sil">{ears}<path d="{head}"/></g>
  <mask id="faceA"><rect x="-20" y="-20" width="140" height="140" fill="white"/>{face}</mask>
  <mask id="faceB"><rect x="-20" y="-20" width="140" height="140" fill="white"/>{face}</mask>
  <mask id="behind"><rect width="140" height="100" fill="white"/>
   <g transform="{B}"><use href="#sil" fill="black" stroke="black" stroke-width="5"/></g>
   <path d="{rim}" fill="none" stroke="black" stroke-width="12"/>{pawcut}
  </mask>
  <mask id="front"><rect width="140" height="100" fill="white"/>
   <path d="{rim}" fill="none" stroke="black" stroke-width="12"/>{pawcut}
  </mask>
  <mask id="mound"><rect width="140" height="100" fill="white"/>{pawcut}</mask>
 </defs>
 <g mask="url(#behind)"><g transform="{A}"><g mask="url(#faceA)"><use href="#sil"/></g></g></g>
 <g mask="url(#front)"><g transform="{B}"><g mask="url(#faceB)"><use href="#sil"/></g></g></g>
 <path mask="url(#mound)" d="M8,89 C8,79 36,75 70,75 C104,75 132,79 132,89 C132,92 130,94 127,94 L13,94 C10,94 8,92 8,89 Z"/>
 {pawfill}
 <path d="M70,15.5 C70,11.5 64.5,10.5 64.5,14.6 C64.5,18 70,21.5 70,21.5 C70,21.5 75.5,18 75.5,14.6 C75.5,10.5 70,11.5 70,15.5 Z"/>
</svg>'''
if __name__ == "__main__" and "hug" in sys.argv:
    open("fam-hug.svg", "w").write(hug())

# The search prompt with linked Macs: the family's search mole, unchanged in size and place, with a
# smaller one leaning in beside it for each linked Mac (one on the left, or one on each side for two
# or more). The small ones lean from the hips - head and body rotate, the mound stays level - the
# way the hug's moles do. 176x100, so at 68 pt tall the centre mole is the plain prompt's mole.
STAR_EYES = None
def mole(pid, eyes, lean=0.0, sparkles=""):
    """One family mole in its own 100x100 frame, with ids prefixed so several share an SVG."""
    paws = [(33, 71), (67, 71)]
    pawcut = "".join(f'<ellipse cx="{x}" cy="{y}" rx="10" ry="7.2" fill="black"/>' for x, y in paws)
    pawfill = "".join(f'<ellipse cx="{x}" cy="{y}" rx="7.6" ry="5"/>' for x, y in paws)
    rot = f'rotate({lean} 50 90)'
    defs = f'''
  <mask id="{pid}face"><rect x="-30" y="-30" width="160" height="160" fill="white"/>
   {eyes}
   <ellipse cx="50" cy="49.5" rx="6.8" ry="4.9" fill="black"/>
   <rect x="45.8" y="56" width="8.4" height="6.4" rx="1.6" fill="black"/><rect x="49.35" y="55" width="1.3" height="9" fill="white"/>
  </mask>
  <mask id="{pid}level"><rect x="-30" y="-30" width="160" height="160" fill="white"/>
   <path d="M6,88 C6,76 26,70 50,70 C74,70 94,76 94,88" fill="none" stroke="black" stroke-width="12"/>{pawcut}
  </mask>
  <mask id="{pid}mound"><rect x="-30" y="-30" width="160" height="160" fill="white"/>{pawcut}</mask>'''
    body = f'''
  <g mask="url(#{pid}level)"><g transform="{rot}"><g mask="url(#{pid}face)">
   <circle cx="31" cy="25.5" r="4"/><circle cx="69" cy="25.5" r="4"/>
   <path d="M25,90 L25,45 C25,27 36,18 50,18 C64,18 75,27 75,45 L75,90 Z"/>
  </g></g></g>
  <path mask="url(#{pid}mound)" d="M9,89 C9,79 27,75 50,75 C73,75 91,79 91,89 C91,92 89,94 86,94 L14,94 C11,94 9,92 9,89 Z"/>
  {pawfill}{sparkles}'''
    return defs, body

def search_linked(n):
    eyes = f'<path d="{star(40,39,6.2)}" fill="black"/><path d="{star(60,39,6.2)}" fill="black"/>'
    main_d, main_b = mole("m", eyes, 0, f'<path d="{star(84,20,6)}"/><path d="{star(92,32,3.4)}"/>')
    parts = [("m", main_d, f'<g transform="translate(38,0)">{main_b}</g>')]
    # Half size, standing on the same ground line (the mound's bottom, y=94), centred 21 from the edge.
    side = [("l", 8, 21 - 25)] + ([("r", -8, 176 - 21 - 25)] if n >= 2 else [])
    for pid, lean, tx in side:
        d, b = mole(pid, eyes, lean)
        parts.append((pid, d, f'<g transform="translate({tx},47) scale(0.5)">{b}</g>'))
    defs = "".join(d for _, d, _ in parts); body = "".join(b for _, _, b in parts)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 176 100">\n <defs>{defs}\n </defs>{body}\n</svg>'

if __name__ == "__main__" and "linked" in sys.argv:
    open("fam-search-linked1.svg", "w").write(search_linked(1))
    open("fam-search-linked2.svg", "w").write(search_linked(2))
