# dmgbuild settings for the Omni installer DMG. Lays out the classic drag-to-install window
# (app icon + arrow -> Applications) by writing the .DS_Store directly - no Finder/AppleScript, so
# it works headless on the self-hosted CI runner without GUI automation permissions.
#
#   dmgbuild -s Scripts/dmg/settings.py -D app=<Omni.app> -D bg=<background.png> [-D extra=<file>] \
#            "Omni <version>" dist/Omni-<version>.dmg
import os.path

app = defines["app"]
bg = defines["bg"]
extra = defines.get("extra", "")            # optional first-launch help text (un-notarized builds)
appname = os.path.basename(app)

format = "UDZO"                              # compressed, read-only
files = [app] + ([extra] if extra else [])
symlinks = {"Applications": "/Applications"}

background = bg
default_view = "icon-view"
icon_size = 110
text_size = 13
window_rect = ((360, 220), (660, 420))      # (x, y), (w, h) in points

# Icon centers (points, origin top-left) - aligned with the arrow drawn in the background.
icon_locations = {
    appname: (165, 205),
    "Applications": (495, 205),
}
if extra:
    icon_locations[os.path.basename(extra)] = (330, 350)

# A clean installer window: no toolbar/sidebar/status/path bars.
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
