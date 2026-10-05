# The film's asset kit: one prompt per plate or sprite, generated in parallel, sprites cut out.
# usage: assets.py [name ...]   (no names: everything missing)
import sys, os, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "work", "launch", "assets")
MOLE = os.path.join(HERE, "..", "work", "launch", "style", "b-mole.png")
ISO = (" Isolated single object centered on a plain seamless cream paper background, nothing else in "
       "the frame, generous margin around it, soft contact shadow only.")
MOLE_REF = ("The mole character from the reference image, the same cut-paper craft figure with layered "
            "brown cardstock fur, same face, eyes, nose and buck teeth. ")

# name: (prompt, aspect, size, refs, cut)
A = {
    # backdrops
    "bg-paper": ("An empty sheet of warm cream cardstock filling the whole frame, photographed straight "
                 "on, very subtle paper fiber texture, soft light falloff toward the corners. Nothing else.",
                 "16:9", "4K", [], False),
    "bg-tall": ("A tall vertical cut-paper cross-section. Top half: an empty pale cream paper sky with a few "
                "faint layered paper cloud wisps at the very top edge. A crisp horizontal ground line at the "
                "middle, with a thin strip of kraft paper grass. Bottom half: many horizontal strata of layered "
                "paper soil getting darker with depth, kraft, tan, brown, deep brown, with a few small pebbles "
                "and roots, and at the very bottom a cozy round burrow chamber glowing softly with teal light.",
                "9:16", "4K", [], False),
    "bg-strata": ("A wide empty underground chamber in cut paper: horizontal strata of layered paper soil in "
                  "kraft and brown tones framing a large, softly lit, empty cavity in the middle, warm light. "
                  "Calm, lots of empty space in the cavity for things to be placed later.",
                  "16:9", "4K", [], False),
    # props
    "cloud": ("A large puffy cloud built from many overlapping layered paper circles, only white and cream "
              "paper, no dark or brown pieces, seen from the front." + ISO, "16:9", "2K", [], True),
    "doc": ("A single blank sheet of cream paper standing upright, slightly curled, with a folded top-right "
            "corner, seen straight from the front, no writing on it." + ISO, "1:1", "2K", [], True),
    "photo": ("A single blank instant-photo print made of paper with a cream border and a soft teal and "
              "kraft paper-cut landscape in the picture area, seen straight from the front." + ISO,
              "1:1", "2K", [], True),
    "folder": ("A single closed kraft paper file folder with a tab, seen straight from the front." + ISO,
               "1:1", "2K", [], True),
    "tag": ("A single blank kraft paper luggage tag with a reinforced hole and a short loop of cotton "
            "string, hanging straight down, seen from the front, no writing." + ISO, "1:1", "2K", [], True),
    "card": ("A single blank cream paper index card hanging from a thin thread, seen straight from the "
             "front, no writing." + ISO, "1:1", "2K", [], True),
    "scissors": ("A pair of open scissors crafted from layered paper and cardstock, teal handles, side "
                 "view, blades pointing left." + ISO, "1:1", "2K", [], True),
    "cable": ("A single horizontal cut-paper cable, a long thin strip of rolled cream paper with a small "
              "paper plug at the right end, perfectly straight and horizontal." + ISO, "16:9", "2K", [], True),
    "desk": ("A small wooden-looking paper craft desk with a matching chair and a closed silver paper "
             "laptop on top, side view." + ISO, "1:1", "2K", [], True),
    "laptop-open": ("A paper craft laptop, open, seen perfectly straight from the front at eye level, the "
                    "screen a large blank flat pale panel facing the camera, symmetric, no perspective." + ISO, "16:9", "2K", [], True),
    "hands": ("Two cut-paper hands in soft tan cardstock typing on a paper laptop keyboard, close side "
              "view, no faces, no people visible beyond the hands." + ISO, "16:9", "2K", [], True),
    "seal": ("A round wax-seal style rosette made of layered teal and cream paper with a ribbon tail, "
             "blank center, no text." + ISO, "1:1", "2K", [], True),
    # the mole
    "mole-pop": (MOLE_REF + "Popping up out of a round burrow hole with both paws raised in a cheerful "
                 "hello, mid-motion, paper crumbs flying." + ISO, "1:1", "2K", [MOLE], True),
    "mole-dig": (MOLE_REF + "Full body, side view facing right, digging busily with its front claws, a "
                 "little arc of paper soil crumbs behind it." + ISO, "1:1", "2K", [MOLE], True),
    "mole-carry": (MOLE_REF + "Full body, side view facing right, walking and carrying a long thin strip "
                   "of cream paper over its shoulder." + ISO, "1:1", "2K", [MOLE], True),
    "mole-listen": (MOLE_REF + "Full body, sitting, looking straight up with one paw cupped to its ear, "
                    "listening carefully." + ISO, "1:1", "2K", [MOLE], True),
    "mole-wave": (MOLE_REF + "Full body standing, waving goodbye with one paw, friendly smile, front "
                  "view." + ISO, "1:1", "2K", [MOLE], True),
}

def make(name):
    prompt, aspect, size, refs, cut = A[name]
    png = os.path.join(OUT, name + ".png")
    if not os.path.exists(png):
        cmd = [sys.executable, os.path.join(HERE, "gen.py"), png, prompt, "--aspect", aspect, "--size", size]
        for r in refs: cmd += ["--ref", r]
        subprocess.run(cmd, check=True, capture_output=True)
    if cut and not os.path.exists(os.path.join(OUT, name + "-cut.png")):
        subprocess.run([os.path.join(HERE, "..", "work", "launch", "cutout"), png,
                        os.path.join(OUT, name + "-cut.png")], check=True, capture_output=True)
    return name

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(A)
    with ThreadPoolExecutor(8) as ex:
        for n in ex.map(make, names): print("done", n, flush=True)
