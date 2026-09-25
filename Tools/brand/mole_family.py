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
    elif v == "hush":
        # one brow up, one eye narrowed, a paw's finger over the mouth: nothing to tell yet
        eyes = ('<path d="M36.6,39.6 L43.4,39.6 A3.4,3.4 0 0 1 36.6,39.6 Z" fill="black"/>'
                '<circle cx="60" cy="39" r="3.6" fill="black"/>'
                '<path d="M34.6,32.4 L45,35.2" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>'
                '<path d="M54.6,31 Q59.5,25.2 65,29.6" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round"/>')
        # the other paw comes up under the chin, and its finger stands over the mouth
        paws = [(33, 71), (50, 70)]
        extra_mask = '<rect x="45.3" y="54.8" width="9.4" height="16" rx="4.7" fill="black"/>'
        extra_fill = '<rect x="47.6" y="57" width="4.8" height="12" rx="2.4"/>'
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
for v in ["base", "search", "ocr", "sleep", "serious", "empty", "hush"]:
    open(f"fam-{v}.svg", "w").write(svg(v))
