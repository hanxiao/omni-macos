# UI: native macOS look and behaviour

Moved out of CLAUDE.md on 2026-10-03, section text unchanged except where marked SUPERSEDED or
UPDATED. Dated findings, measurements and rejected options: read the section before touching
the code it names.

## Filter chips in the search field

- THE PLATFORM HAS THIS CONTROL: `searchable(text:tokens:placement:prompt:token:)`, macOS 13+,
  so no token-field package and no `NSTokenField` wrapping. Filters are chips in the field and
  the box holds only the semantic text. The chips are a projection of the canonical query string,
  which is still the one thing history and back/forward replay.
- TOKENS MUST BE STORED, NOT COMPUTED. The field mutates the collection it is bound to and keeps
  state beside it, so a getter that rebuilt the array from `activeQualifiers` on every read
  desynchronised it and the chips vanished the moment anything was typed after them.
  `searchTokens` is stored and `syncSearchTokens()` keeps it in step at the three places
  `activeQualifiers` is assigned.
- ONLY A REMOVAL IS A USER EDIT. The field writes the collection back on its own account as it
  re-renders; treating an echo as an edit rebuilt the query from stale state. `setSearchTokens`
  ignores anything that is not strictly shorter.
- PROMOTE A QUALIFIER ONLY WHEN THE FINISHED WORD IS ONE. Re-parsing on every space also
  re-normalises the text, which swallowed the space itself: "a very long" typed straight through
  arrived as "averylong". The edit path now re-parses only when a trailing space follows text
  that actually contains a qualifier, and Return promotes whatever is left. Mid-word, `type:i`
  stays editable text - promoting it would build a chip out of half a word.
- THE TOOLBAR SEARCH FIELD WILL NOT GROW ON TAHOE. Measured at 0/1/2/3 chips it is 330 px every
  time, with `NSSearchToolbarItem.preferredWidthForSearchField` set (it reads back the value and
  changes nothing) AND with a width constraint on `item.searchField`. The Liquid Glass toolbar
  group lays its contents out itself. The sizing code was written, measured to do nothing, and
  deleted; do not write it again.
- A TOKEN CHIP RENDERS ITS TITLE ONLY - SwiftUI drops the `systemImage` from a token's `Label` on
  macOS, measured. So the KEY has to be in the text (`type:image`, not `image`), or a chip cannot
  be told from a tag of the same name. No space after the colon and paths show their last
  component, because the width is fixed and every character costs one of the query's.
- The qualifier bar under the toolbar is gone except in plain-text mode, where it explains why
  the chips do not apply and offers the way back. Entering that mode moved to the filter menu.

## Folder browsing (App/FolderBrowser.swift)

- SELECTING A FOLDER BROWSES IT, it does not draw a map. The embedding map said nothing about
  what is IN a folder and left the search unscoped; the browser lists the folder Finder-style,
  folders before files, and a double-click descends. The map is still there, behind "Show folder
  map" in the folder's context menu, which clears `filterFolder` because the two share the
  empty-result region and the browser wins.
- THE BROWSED FOLDER IS `filterFolder`. There is no second "current directory" and there must not
  be, because that one property already did everything and simply had no UI reaching it: the store
  filter takes `folderPrefix` so a search is ALREADY scoped to the subtree, `syncBoxFromFilters`
  already writes `in:"<path>"` into the box, and a `NavEntry` carries that box string - so
  back/forward walks folders with no new history. Connecting those was the whole feature.
- `hasQuery` IS WHAT MAKES IT WORK. It reads the SEMANTIC query, not the raw box, so setting a
  folder filter fills the box with `in:"..."` without counting as a query - which is what keeps
  the empty-result region (and so the browser) available. Typing hides the browser instantly and
  the results are already scoped. Do not "fix" `hasQuery` to read `rawQuery`.
- A sidebar selection that is NOT a folder must not clear `filterFolder`. A history row applies
  its own filters first, and clearing afterwards wiped them straight back out; the else branch
  only clears the map.
- THE BROWSED FOLDER'S QUALIFIER CHIP IS SUPPRESSED while the browser is up. The chips in
  `QualifierBar` are display-only - `Text(key)` + `Text(value)` in a capsule, no action - so a
  chip of the path the breadcrumb is already showing is a second copy that does strictly less.
  Other qualifiers still get chips; the bar hides entirely when nothing is left to show.
- THE VIEW TOGGLE HAD TO BE LET OUT OF ITS GATE. Sort and view appear only once there are
  results ("only meaningful with results"), so the browser's gallery view existed and NOTHING
  could switch to it - written, built, unreachable. The condition now includes the browser, and
  the browser honours `sortOrder` for real: folders before files, then Name or Date modified,
  with `.relevance` reading as Name because it means nothing for a directory.
- No separators between rows: Finder's list view draws none, and a folder listing is not a table.
- THE GALLERY SHOWS REAL THUMBNAILS, via the app's existing `Thumbnail` view - the same
  QuickLook-backed, memory-bounded cache the results list uses, which already falls back to the
  type icon when QuickLook has nothing. A gallery of type icons is not a gallery. Folders keep a
  folder icon and skip the well and border so they do not read as boxed files, and that icon is
  fetched ONCE: `NSWorkspace.icon(forFile:)` calls the icon daemon per call, and a directory of a
  few hundred folders would ask it a few hundred times for the same picture.
- Packages (.app, .rtfd) are files, not folders - `isPackageKey`, the way Finder treats them.
  Directory listing runs off the main thread: a home folder is not a frame's worth of work.
- Verified in the app, not just built: sidebar click browses, double-click descends, the
  breadcrumb walks back up, `readme` inside 911-fanbook returns only that folder's files, and
  back/forward moves between folders with the chevrons enabling and greying correctly.
- It lists what the INDEX knows, never the directory. `VectorStore.indexedChildren(ofFolder:)`:
  files indexed directly in the folder, plus subfolders holding at least one indexed file beneath
  them, so descending can never dead-end. Checked against the live index - ~/Documents has 58
  files + 94 folders on disk and 51 + 88 in the index, and the six hidden folders are genuinely
  unindexed. Dates and sizes still come from DISK, the way Finder reports them; the index's copies
  belong to the version that was indexed and go stale on the next edit. A path that no longer
  stats is dropped rather than listed, which makes a deleted file self-heal out of the listing.
- That query is SQL over `dirs`, and it has to be. The obvious version - one pass over the
  in-memory `idPath` table, the way `fileCount(underFolder:)` does it - is O(live files), 2.6M
  here, and it held the browser on a spinner for MINUTES in a debug build (the main thread was
  idle the whole time; `sample` showed the work on a background thread, so it never looked like a
  hang). `dirs` has one row per directory and a unique index on `path`, so the range scan
  `>= folder||'/'` .. `< folder||'0'` (the `StoreSchema.dirSubtreeIDs` idiom; '0' is the byte
  after '/') plus EXISTS clauses riding `files(dir_id, name)` and `chunks(file_id, chunk_index)`
  make it instant.
- No trailing chevron on folder rows. Finder's LIST view has none - it puts a disclosure triangle
  on the LEFT - and a right chevron is its COLUMN view's idiom, not this one.

## Liquid Glass (App/Design.swift) - audited 2026-09-11 against the macOS 26.2 SDK

Adoption is complete for what this app is. Verified running on Tahoe, not read off a doc: the
score chip over a dark synthwave thumbnail renders dark with the magenta behind it showing
through, and over a bright beach photo the same chip renders light and warm. A static
`.ultraThinMaterial` would look identical on both. That is the real `glassEffect` lensing.

Preconditions, all checked: built against `macosx26.2`, `LSMinimumSystemVersion` 14.0, and NO
`UIDesignRequiresCompatibility` / `NSRequiresAquaSystemAppearance` opt-out anywhere - so every
standard control gets Liquid Glass automatically and only custom surfaces need the API.

In use: `glassEffect(_:in:)` with `.regular` and `.regular.interactive()`, `GlassEffectContainer`
(via `GlassGroup`), `ToolbarSpacer`, `sharedBackgroundVisibility`, and a Reduce Transparency
fallback to `.ultraThinMaterial` - which is both the HIG-correct behaviour and a real GPU saving,
since it removes a live glass pass per visible cell.

GALLERY BADGES ARE A MATERIAL, NOT GLASS (2026-09-23, reverses the note below that called them "the
one legitimate in-content use"). The HIG says "Don't use Liquid Glass in the content layer", and a
grid of glass badges was one GlassEffectContainer per visible cell, each re-sampling on scroll.
`mediaBadge()` is `.thinMaterial`. Glass stays for controls that float over content: the OCR readout,
find bar, notice chip, map overlays. Also: the bars in `TopBar` have no fill or rule on Tahoe (a
background behind a scroll-edge bar blocks the soft edge, WWDC25 323); shadows on glass chips are
drawn only in the material fallback (`chipShadow`); the back/forward capsule is `.interactive()`;
drop rings are `ConcentricRectangle` on Tahoe (`DropRing`).

Deliberately NOT used, so nobody "fixes" this later:
- `.buttonStyle(.glass)` / `.glassProminent` - every button we would apply it to already sits
  INSIDE a glass chip, and glass inside glass is the thing Apple tells you not to do. Standard
  `.bordered` / `.borderedProminent` buttons (onboarding, settings) already get the new look for
  free by recompiling.
- `Glass.clear` - Apple recommends it over media-rich content, but only with a dimming layer.
  `.regular` was checked on both a dark and a bright photo and reads cleanly on each; `.clear`
  would trade that for a dimming layer we would then have to tune.
- `backgroundExtensionEffect()` - wants full-bleed media running under a sidebar. The detail pane
  is a results grid.
- `scrollEdgeEffectStyle(_:for:)` - WRONG WHEN FIRST WRITTEN, and worth the correction: this said
  Finder's icon view clips hard at the toolbar boundary the way ours did. It does not. Both its
  list and its icon views use the SOFT pocket - measured, a gradient from 245 to 253 over ~25pt with
  no rule. Ours had the hard one. The modifier still turns out not to be the fix: see the
  scroll-pocket note under "Folder browser columns".
- `Glass.tint(_:)`, `Glass.identity` - no surface wants them.

Unused and arguably worth it, if the subject ever comes up again: `glassEffectUnion(id:namespace:)`
would make the gallery cell's score / locator / stack chips ONE glass shape rather than three that
merge when they approach; `glassEffectID(_:in:)` plus `glassEffectTransition(.materialize)` would
let chips morph and form instead of sliding in on a generic `.move` transition; and
`ConcentricRectangle` / `containerShape` would keep nested corner radii concentric instead of the
fixed `Design.corner = 8`. All three are refinements, none is a defect, and none has been done.

## Third-party UI packages (surveyed 2026-09-11)

The app has TWO dependencies: `mlx-swift` and `swift-tokenizers`. `App/Updater.swift` says it in
one line - "No Sparkle / no third-party dependency." A package has to close a gap the system API
does not, and has to be safe for an Apache-2.0 notarised public app on macOS 14. Most are not.
This survey exists so the search is not repeated.

- BUILT AND REMOVED, 2026-09-11: a global summon shortcut (`sindresorhus/KeyboardShortcuts`, MIT,
  Carbon `RegisterEventHotKey`), a floating `NSPanel` quick-search palette over the shared
  `AppModel`, and a menu bar item. All three worked end to end - Option-Space from another app
  raised the palette, typing searched, Return opened, Command-Return revealed in Finder,
  Shift-Return handed the query to the main window. Han looked at it and said no: "I don't like
  menu icon and call out search popup thing." Omni is a window, not a launcher. Do not re-propose
  this. The dependency went back out with it, and the count is two again.
- REJECT `siteline/swiftui-introspect` (MIT, 6.5k, active, macOS 12). Good software, wrong trade:
  nine AppKit spelunking sites in `App/` is too few to justify a dependency whose entire business
  is tracking undocumented SwiftUI view hierarchies. It breaks on macOS betas by design.
- REJECT `mrkai77/Luminare` on licence alone. GitHub reports "Other" and there is no resolvable
  LICENSE on main. An Apache-2.0 public app cannot take that. Same gate as STTextView.
- REJECT `SwiftUIX/SwiftUIX` (MIT, 8.2k). An "exhaustive expansion" of SwiftUI: the cost is the
  whole package, the benefit is a handful of views, and its macOS 11 floor means its shims predate
  everything we ship on.
- REJECT `EmergeTools/Pow` (MIT, 4.4k, last release 2024-11, iOS-shaped). The honest version of
  that want was unused SYSTEM API, and it has now been spent - see below.
- REJECT `sindresorhus/Settings` - the app already uses the SwiftUI `Settings {}` scene.
  REJECT `sindresorhus/Defaults` - 75 working `UserDefaults` sites; a typed refactor with nothing
  user-visible at the end. REJECT `LaunchAtLogin-Modern` - it wraps the one `SMAppService` call
  that `App/LaunchAtLogin.swift` now makes directly, and has not been touched since 2023-12.
- OPEN `nalexn/ViewInspector` (MIT, 2.6k, macOS 10.15). Worth a spike, not a commitment: it
  unit-tests SwiftUI view trees IN PROCESS, the one thing that would unblock UI testing without
  the XCTest automation password prompt. It reflects into SwiftUI's private view storage, same
  fragility as introspect - but test-only, so a break cannot reach a user. That asymmetry is the
  whole argument. Last tag 0.10.3 (2025-09-21), 60 open issues, `@Observable`-era support
  unverified.

### MenuBarExtra keeps its content live (measured, 2026-09-11)

A SwiftUI `MenuBarExtra`'s content stays in the view graph whether or not the menu is open, so
reading anything from `AppModel` in it puts the whole MAIN menu on that property's change rate.
With a status line reading `model.progress.perRoot`, `sample` showed 100% of the main thread
inside `AppDelegate.scenesDidChange -> makeMainMenu -> updateMenuHost -> requestUpdate`, looping:
the app never reached idle, a `DispatchQueue.main.asyncAfter(5s)` never fired, and the global
hotkey - correctly registered, `RegisterEventHotKey` probe returned `-9878` - could not be
delivered. The same shape as the OCR rail stutter: a high-frequency write invalidating a whole
tree that only needs to be correct when someone looks at it. A static `MenuBarExtra` costs
nothing; `NSStatusItem` with `menuNeedsUpdate` costs nothing and never touches the main menu. If a
menu bar item is ever wanted again, that is the way to build it.

### SwiftUI will not give you a plain Return (measured, 2026-09-11)

For a focused `TextField`: `onSubmit` sees ONLY an unmodified Return, `onKeyPress` on an ancestor
sees only a MODIFIED one, and once that `onKeyPress(keys: [.return])` exists it takes the plain
Return too and drops it - so the combination leaves Return doing nothing at all. Tab never arrives
either; focus traversal eats it first. A local `NSEvent` monitor runs ahead of the responder chain
and is the only place all the chords can be decided together.

### Motion: system API, not a package

`.contentTransition(.numericText(value:))` and `.symbolEffect` are macOS 14 and were the answer to
wanting Pow. Live counters roll their digits instead of cutting: the OCR chip's page count and
tok/s (`App/OCRView.swift`, `ProgressReadout`) and the empty state's indexed-file count
(`SearchWaysPrompt.count`). Pass the NUMBER to `numericText(value:)`, not just the string - that
is what makes a rising count roll up and a falling rate roll down.

## Settings window, audited pane by pane (2026-09-11)

Seven tabs is the ceiling at `width: 480`. An eighth (a "General" tab, since removed) pushed
SERVING behind an overflow chevron, where a whole pane only opens as a popover outside the window.
If a tab is ever added, one has to go or the window has to get wider - check the strip, the
overflow is silent.

Fixed in this pass:
- `ByteCountFormatter.string(fromByteCount:countStyle:)`, the static convenience, leaves
  `allowsNonnumericFormatting` ON and renders 0 as "Zero KB". That shipped in the memory legend.
  No Apple surface says it - Finder's Get Info says "0 bytes". All byte formatting now goes through
  `ByteSize` in App/Design.swift, which turns that off once; it is `@MainActor` because
  `ByteCountFormatter` is not `Sendable` and the instances are cached rather than rebuilt per row.
- Labels are sentence case throughout this window ("Search as you type", "Max image size").
  "Benchmark This Mac" and "Add searches to History" were title case and are not any more. Button
  labels stay as they are - "Restore Default", "Reveal in Finder" - which is the macOS convention.
- The Serving port was a bare titled `TextField` with `.frame(width: 90)`. A width-constrained
  field hugs its own label, so the number sat right after the word "Port" while every other row in
  the window puts its value at the trailing edge. `LabeledContent` plus trailing text alignment
  puts it back on the column.

## Four search-box fixes (2026-09-12)

- ESCAPE CLEARS THE WHOLE QUERY. `AppModel.clearSearch()` goes through `applyParsedQuery("")`,
  the same door a parse uses, because the chips, the qualifier bar and the store filter are all
  PROJECTIONS of `rawQuery` - clearing the text alone left them behind, so a "cleared" box still
  returned a filtered, empty result set. Bound with `onKeyPress(.escape)` on the searchable
  content, and it returns `.ignored` when there is nothing to clear so Escape still does its
  normal job everywhere else.

- CLICKING A HISTORY ROW NO LONGER RECORDS A TWIN. `runHistoryQuery` set the re-record guard to
  the item's stored text, then `applyParsedQuery` rewrote the box into canonical form
  (`in:/Users/x model` -> `in:"/Users/x" model`). The guard compared unquoted against quoted,
  never matched, and every click recorded a quoted duplicate. The guard now holds `rawQuery`,
  read AFTER canonicalisation. Verified: 164 items, three clicks, still 164.
  That alone was not enough: the RECORDER deduped on the exact string, so a query TYPED unquoted
  still landed beside the canonical row the box rewrites to. Dedup now runs on
  `HistoryItem.canonicalKey` - parse, lowercase the semantic text, sort the qualifiers - which is
  the same key `AppModel.canonicalized` uses to collapse the twins already on disk at load
  (newest `lastUsed` wins, a bookmark on either survives). Verified end to end: the parser returns
  identical output for both spellings, the load merge took 167 rows to 164 and then 165 to 164,
  and three history clicks left the count at 164.

- A CLICK IN THE FOLDER BROWSER SELECTS, AND LOOKS LIKE FINDER. A tap gesture of ANY kind on a row
  swallows the click `List(selection:)` needs - verified twice, including with a double-tap alone,
  which also selected nothing - so both taps and the highlight are ours. Finder's treatment
  exactly: the whole row filled with the ACCENT colour and every label turned white. The first
  attempt used `Color.accentColor.opacity(0.18)` with dark text and read as a pale blue band with
  margins, which is not what Finder draws. Same in the photo browser.

- SUGGESTIONS ARE ONE LINE, AND THE SCOPE IS A CHIP. They used `HistoryItem.displayText` as the
  label, so a 340-character path rendered as a ten-line wrapped paragraph, five of them stacked
  taller than the window. The row is now `lineLimit(1)`, the label is `displayLabel`, and the
  folder is drawn as a chip; the COMPLETION is still the full query, so selecting one replays
  everything. `displayScope` middle-elides through `SearchToken.elided` for the same reason the
  field's chip does - sibling folders here differ only in their suffix.

  ON THE CHIP'S LOOK, since it will be asked again: native macOS suggestion lists (Spotlight,
  Safari, Finder) have NO chips - plain text rows. The chips are a deliberate departure, made
  because these queries carry paths that are illegible as plain text. Given that, the most native
  option available is to match the token the SEARCH FIELD itself draws for the same qualifier, so
  field and suggestions speak one language: a rounded rect, not a capsule, `.primary.opacity(0.06)`
  fill, primary text. `.quaternary` was the first try and measured far heavier than the system
  token (which is roughly a 3% wash on its own surface) - it read as a grey block.
  It is NOT a glass effect on purpose: the suggestion popover is already a vibrant surface, and
  glass inside glass is the one thing Apple's guidance rules out.

HARNESS NOTE, cost an hour twice: `CGEventKeyboardSetUnicodeString` typing into this app is
intermittent - it silently drops the whole string and looks exactly like a broken text field.
`osascript -e 'tell application "System Events" to keystroke "..."'` is reliable. Rule out the
harness before believing a typing regression.

## Sidebar history: labels, and what the buckets really hold (2026-09-12)

Rows show the WORDS, not the query. `HistoryItem.displayLabel` returns the parser's
`semanticText`, so `cat in:/Users/hanxiao/Documents/embedding-inversion type:image` reads as
`cat`. The path was the noise - the same path on every row, the one part that says nothing about
which search it was. A pure-filter search with no words falls back to the qualifier VALUES
("Documents, image"), never the raw string.

Two things had to come back after that, because clean went too far:
- `displayScope` - the `in:` folder's LAST COMPONENT, dimmed, after the label. Without it the same
  word searched in eight folders is eight rows reading "model". The path was the noise, the folder
  was not.
- a generic `line.3.horizontal.decrease` glyph on any filtered row (`tag:` keeps its own). With
  the qualifiers gone a filtered search and a bare one were identical.

`displayText` is UNTOUCHED and must stay so: `id` is derived from it, so it carries identity,
dedup and replay. Only the label changed. The tooltip still shows the whole query.

BUCKETING IS CORRECT - measured, not assumed. Decoded the persisted `omni.searchHistory` (164
items) and re-bucketed it on calendar days the way `historyGroups` does: Today 8, Yesterday 9,
Previous 7 Days 32, Previous 30 Days 111, Earlier 4, spanning 2026-08-13 to 2026-09-12. Dates
survive an upgrade intact - they are plain reference-date doubles in JSON, nothing reseeds them.
What LOOKS like a bug is the shape of the ladder: Today and Yesterday are one day each while
"Previous 30 Days" is 23 days wide and holds 68% of the list, so after any gap the sidebar is
mostly that one section. If this is ever revisited, the fix is the ladder (or a bound on how many
rows a section shows), not the counting.

## The `in:` chip in a deep folder (2026-09-12)

Depth was already handled - the chip shows the folder's LAST COMPONENT, not the path - so a deep
folder is not what breaks it. A long NAME is. This index really holds
`defense_yiyic_sentiment140_1k_google-bert_bert-base-multilingual-cased_train1000_noise_0_epsilon0.05_delta0.0001`,
112 characters, with six siblings differing only after character 90.

SwiftUI does truncate a token that will not fit, but it truncates at the TAIL, so all six siblings
rendered as the identical chip `in:defense_yiyic_sentiment...` and there was no way to tell which
folder the search was scoped to. `SearchToken.elided` now cuts the value in the MIDDLE at 24
characters, in the STRING - not via `truncationMode(.middle)` on the token's `Text`, because a
token chip already drops the `systemImage` off a `Label` and its content modifiers are not
something to rely on. Reads `in:defense_yiyi...delta0.0001` against
`in:defense_yiyi..._delta1e-06`, and the typed query still has room beside it.

Only `label` is elided. `queryText` - what history, back/forward and the store filter all replay -
still carries the full path, and must.

## Folder browser columns (App/BrowserColumns.swift, 2026-09-12)

Finder's header, with the index's facts instead of the filesystem's. Right-click the header to
choose columns, click a title to sort, click again to reverse.

Offered: Kind (the modality the index filed it under - the SAME vocabulary as the filter menu),
Date Modified, Date Added, Date Indexed, Size, Files Indexed (folders), Tags (media). Default on:
Kind, Date Indexed, Size, Files Indexed. Name is not in the menu, because Finder will not let you
turn its Name column off either.

FILES INDEXED is the one column Finder cannot have, and the reason the header is worth building:
it answers "how much of this folder is searchable" at a glance. It comes from a grouped aggregate
over the same `dirs` range scan the listing already runs, so it is free.

DATE ADDED IS FIRST INDEX TIME, and it exists as of v5 - this note used to say it could not. The
schema kept ONE `indexed_at` stamp per file and a reindex overwrote it, so what existed was LAST
indexed; `files.first_indexed_at` is a second stamp, written once on the INSERT and deliberately
absent from the upsert's `DO UPDATE` list. That omission IS the mechanism, and nothing that
compiles would notice if someone added it back, which is what `FirstIndexedTests` is for.

An index written before the column has it added and SEEDED FROM `indexed_at` on the one open that
adds it - exact for any file not re-indexed since, an upper bound for the rest, and strictly
better than the 0 the ALTER's default leaves. The pair runs in ONE transaction: as two statements
the ALTER commits, a kill inside the 4377 ms UPDATE leaves the column present and empty, and the
"already added" guard then reads that as done and the index carries zeros for life. Measured on
the real 2,678,916-file index: 4377 ms, once, on the same open that starts the v5 migration.

A folder row shows the OLDEST stamp beneath it, against the newest for Date Indexed - the pair
says when the folder started being covered and when it was last touched.

FIX THESE BY SCREENSHOTTING FINDER AND OMNI STACKED, NOT FROM MEMORY, AND MEASURE THE PIXELS.
Every detail below was invisible until the two windows were one above the other at the same size
browsing the same folder, and several of them were invisible even then until a column of pixels was
averaged. Finder at {0,40,1300,420}, Omni at {0,470,1300,380}, crop the same strip from each, paste
them together; then scan rows for hairlines and dark runs for text positions. Eyeballing a stacked
capture found the first five; only measurement found the rest.

What that caught, in order:
- the title sat INSIDE the chevrons' Liquid Glass capsule. Tahoe folds consecutive `.navigation`
  items into one pill, so it looked like a third button.
- a hairline ran between the toolbar and the column header. `w.titlebarSeparatorStyle = .none` was
  the first guess and is NOT what draws it - see the scroll-pocket note below.
- the title was near-black; Finder's is a mid grey (sampled: darkest pixel 160 on a 255 ground).
- the sorted column needed weight as well as colour - `.semibold` - because colour alone did not
  read next to Finder's.
- it was a `Menu` with a disclosure chevron. Finder's title is plain text.
- the name did not move when the chevrons were not there. Two `.navigation` items with the empty
  one hidden still cost a full Tahoe inter-group gap on BOTH sides: the name sat at x=405 with a
  2pt item at x=378 in front of it, against x=361 with that item gone. See the single-item note.
- header and rows were 9pt out of line, and drifted a further point per column.
- rows repeated every 24pt against Finder's 20.

THE TOOLBAR/HEADER HAIRLINE IS TAHOE'S SCROLL POCKET, and none of the obvious knobs touch it. The
window's `titlebarSeparatorStyle` was already `.none` (verified by reading it back from the live
window) and the rule was still drawn. There is no `NSSplitViewController` in a SwiftUI window, so
`NSSplitViewItem.titlebarSeparatorStyle` has nothing to set. `scrollEdgeEffectStyle(.soft, for:
.top)` was tried on the `List`, on the browser's root and on the whole detail pane, and the rule
survived all three. Dumping the live view tree named it: a 1pt `_NSLayerBasedFillColorView` inside
`NSHardPocketView < NSScrollPocket < NSTitlebarBackgroundView < NSSplitView` - the pocket belongs to
the SPLIT VIEW's titlebar background, not to any scroll view in the subtree, which is why no
modifier in the subtree reaches it. Finder's list AND icon views both use the SOFT pocket (a
gradient from 245 to 253 over ~25pt, no rule). So `WindowTitleHider` hides the rule directly: walk
down at most 8 levels to `NSTitlebarBackgroundView`, then hide any 1pt-tall fill inside it. Shallow
on purpose - it never walks the content pane - and entirely defensive.

ONE TOOLBAR ITEM HOLDS BOTH THE CHEVRONS AND THE NAME. Two items cannot do it, and both ways of
splitting them were tried:
- make the chevron item CONDITIONAL, and when it comes back it is APPENDED - descending into a
  folder put the chevrons to the RIGHT of the name.
- keep it always present but empty, and Tahoe charges a full inter-group gap on either side of it
  (an item whose shared background is hidden is its own group), which is the 44pt hole.
Inside one item the chevrons simply appear and the name slides to meet them. The item draws no
glass, so the name stays outside it - and then the chevrons need a capsule of their own, which a
`ControlGroup` will NOT give: with the item's shared background hidden each of its buttons grew a
capsule and the pair rendered as two circles. Two plain `.borderless` buttons with a `Divider`
between them and one `.glassEffect(.regular, in: .capsule)` around the lot is Finder's shape.
`.borderless` also mutes its label, so an enabled chevron came out as grey as a disabled one; it
needs an explicit `.primary`.

`OMNI_UI_DEBUG=1` + `kill -USR2 <pid>` dumps `/tmp/omni-debug-toolbar.txt`: toolbar item frames and
constraints, the view-controller tree, and every view thinner than 3pt with its ancestor chain.
That last section is what identified the pocket. Use it for layout facts - the shot it also writes
renders SwiftUI layers as garbage.

The folder name lives in the TOOLBAR, after the back/forward chevrons, exactly where Finder puts
it - not on a content row above the listing, which cost a whole row and made the header look like
a second bar. Three corrections were needed to match, all from looking at Finder rather than
guessing:
- it goes AFTER the chevrons, not before.
- it is PLAIN TEXT. A `Menu` was tried, and its disclosure chevron and button chrome are the
  giveaway - Finder's title carries neither. The ancestors already have two ways up (the back
  chevron, the sidebar), and the full path is on hover.
- the header has NO background fill. In Finder the header sits on the same surface as the toolbar
  with no seam; a `.quaternary` band under it read as a separate component bolted on. The only
  hairline divides the header from the ROWS.

HEADER AND ROWS SHARE ONE SET OF NUMBERS (`BrowserMetrics`, `ColumnSlot`). While each carried
padding of its own they sat 9pt apart and drifted a further point per column, and no value lined up
with its title. Measured off Finder: its dividers sat at x=890/1071/1168 in a 1100pt pane, every
header title and every left-aligned value 9pt after its divider, the one right-aligned value (Size)
8pt before the next, rows every 20pt around a 16pt icon, a row's icon 27pt from the pane edge and
its name at 46.

Four things that geometry depends on, none of them guessable:
- HEADER TITLES ARE LEFT-ALIGNED IN EVERY COLUMN, including numeric ones. Sorting Finder by Size
  puts "Size" hard against the column's leading edge while the values under it stay right aligned.
  The header does not follow the column's own alignment.
- THE DIVIDERS AND THE SORT CHEVRON ARE OVERLAYS. As laid-out views each divider stole a point from
  the strip and walked the header off the rows by one more point per column.
- `List` KEEPS 9pt OF HORIZONTAL INSET that no API reports back - with `listRowInsets` zeroed a row
  still began 9pt in from the pane edge. A header drawn above the list has to add the same 9pt by
  hand (`BrowserMetrics.listInset`). Row padding goes INSIDE the row, not in `listRowInsets`.
- 24pt ROWS ARE `defaultMinListRowHeight`, not anything in the row: a 16pt icon with no padding
  still measured 24. Negative row insets did not move it. `.environment(\.defaultMinListRowHeight,
  20)` does.

`.listStyle(.plain)`, not `.inset`: the inset style adds ~12pt of horizontal inset of its own and
draws the alternating bands as inset rounded capsules. Finder's bands are full-bleed and square.

`BrowserColumn.width` is the WHOLE slot, insets included, so one number lays out both strips.

THE SELECTION FILL IS A ROUNDED INSET RECT, not a full-bleed bar. Measured off Finder: the fill is
the full row HEIGHT but runs x=208..1289 in a pane starting at 200 in a 1300pt window - 8pt in on
the left, ~10 on the right. The alternating bands behind it ARE full-bleed; only the selection is
inset. Same shape in the SEARCH RESULTS, whose rows are taller but should not round differently.

THE RADIUS WAS FITTED, NOT PICKED. Reading Finder's fill edge scanline by scanline down from the
top, its inset runs 4,3,2,1,0 px. A 4pt CONTINUOUS corner gave 1,0 - a squircle is much flatter near
the edge than a circular arc of the same radius - and circular 6 gave 3,1,1,0. Circular 8 reproduces
Finder's profile exactly. Measure the profile, do not eyeball the number.

NO WELL BEHIND A THUMBNAIL. `Thumbnail` used to draw a translucent card with a faint stroke around
it, which put a second background behind file icons that already have a shape of their own; Finder
draws neither, in list view or in icon view, and a page of them read as a grid of boxes. The clip
stays, so a photo still gets rounded corners.

THE SEARCH RESULTS LIST IS STRIPED TOO. It cannot use `alternatingRowBackgrounds()` - it is a
`ScrollView` of rows rather than a `List`, deliberately (see the note at `listView`) - so it bands
by index on `NSColor.alternatingContentBackgroundColors[1]`. Its `LazyVStack` spacing went to 0 at
the same time: with a 2pt gap the stripes read as separate cards instead of a ruled table.

The Photos browser uses the same metrics and the same `ColumnSlot` - a Photos source is a folder as
far as browsing goes. Its column header belongs to the LIST, not to the browser: an icon grid has
no columns and Finder draws no header over one.

## The menu bar, diffed against Finder's (2026-09-12)

DUMPED BOTH MENU BARS THROUGH THE ACCESSIBILITY API AND COMPARED THEM ITEM BY ITEM, rather than
from memory - `osascript` over `menu bar items of menu bar 1` for Finder and for Omni. Worth
repeating whenever a feature lands: three of the gaps below were invisible until the two dumps sat
side by side.

- NO GO MENU AT ALL. Finder's is navigation (Back, Forward, Enclosing Folder), then places, then
  the typed path. Ours had Back/Forward buried in View, no way up a level, and no Go to Folder.
  There is a Go menu now with all of it, plus the indexed roots and Photos sources as the "places"
  (Finder lists Documents/Desktop/Downloads there; ours are whatever the user added, which is the
  same idea with the right contents for this app) on Ctrl-Cmd-1..9, since Cmd-1/2 are the view modes.
- BACK/FORWARD MOVED to Go, and were REMOVED from View. They cannot be in both: two items with one
  key equivalent make AppKit strip the chord from one, which reads as Cmd-[ silently doing nothing.
- SHOW SIDEBAR WAS MISSING ENTIRELY ON TAHOE. It was gated to pre-26 on the belief that macOS 26
  supplies its own item in View; the dump shows a View menu with no sidebar item, so the only way to
  unhide the sidebar was the toolbar button. Now unconditional.
- FEATURES THAT WERE CONTEXT-MENU OR TOOLBAR ONLY now have File-menu items: Generate Tags, Search in
  This Folder, Visualize (UMAP/PCA), Ignore Folder, and the Serve over HTTP toggle. A feature
  reachable only by right-click is one most people never find, and it cannot carry a working key
  equivalent either - a chord declared inside a closed context menu never fires on macOS.

GO TO FOLDER COMPLETES AGAINST THE INDEX, not the filesystem. Finder completes against the disk,
which is right for Finder; here the useful answer is narrower - a folder Omni has actually indexed,
because those are the ones that can be browsed and scoped to. It rides the same `dirs` range scan
the browser already uses (`indexedFolders(matching:)`, shortest path first, capped at 12), and a
path that is not indexed is still accepted if it exists, because the empty state is a truthful
answer rather than a refusal. LIKE's own wildcards are escaped - most paths contain `_`, which
would otherwise match any character.

## Closing the window does not quit (2026-09-12)

The red button HIDES the window. Omni keeps serving over HTTP, keeps its MCP endpoint up and keeps
answering skills with nothing on screen, so closing the window is "put it away", the way it is in
Chrome and Mail - not a request to stop the service. Quit is Cmd-Q, the one gesture that means it.

Three parts, and all three are needed: `applicationShouldTerminateAfterLastWindowClosed` returns
false; `isReleasedWhenClosed = false` on the window, so there is something left to bring back (state,
sidebar width and the current search all survive - verified by closing with a sheet open and finding
the sheet still there on reopen); and `applicationShouldHandleReopen` orders it front, which is what
a Dock click, a Spotlight open and `open -a` all go through.

## Toolbar visibility, audited mode by mode (2026-09-12)

Three modes, and every item belongs to a stated one. The rule is that a control is SHOWN only where
it changes something visible - not greyed, not "harmlessly there".

|                          | idle | browse | search | OCR |
|--------------------------|------|--------|--------|-----|
| sidebar toggle, OCR toggle | yes | yes | yes | yes |
| back/forward             | -    | yes    | yes    | **no** |
| name in the toolbar      | -    | folder / source | - | **document** |
| bookmark (star)          | -    | yes    | yes    | -   |
| filter                   | -    | **no** | yes    | -   |
| sort                     | -    | **no** | yes    | -   |
| grid/list                | -    | **yes**| yes    | -   |
| search by file, share    | yes  | yes    | yes    | - (OCR has its own) |
| OCR view / open / save / copy / share | - | - | - | with a document |

The five entries in bold were wrong, and four of them were wrong in the same direction - chrome that
did nothing where it stood:

- SORT was shown while browsing. It orders RESULTS, which are ranked; a browser sorts by clicking a
  column header. Inert there.
- FILTER was shown while browsing, because entering a folder sets `filterFolder` and that makes
  `filtersActive` true. The browser lists indexed children and no filter touches that listing.
- BACK/FORWARD were shown in OCR mode. That trail is the SEARCH history: stepping it while reading
  a transcript changes a result set you cannot see, and the chevrons read as page navigation for
  the document, which they are not.
- GRID/LIST was missing in the Photos browser (the condition named `showsFolderBrowser` only) and
  therefore its gallery existed with no way to reach it.
- OCR mode showed no name at all - the one mode whose toolbar did not say what was on screen.

The star STAYS while browsing, deliberately: bookmarking a browsed folder saves a state you can
return to, which is not the same as a control that does nothing.

THE CHROME IS 44pt TALL, NOT 40. Measured against Finder at the same window size: its chrome runs
y=0..43 and ours ran 0..39, both holding the same 36pt item capsules - so ours had 2pt of air around
them where Finder has 4, and the whole bar read as slightly squashed. `w.toolbarStyle = .unified`
is what Finder uses; hiding the title text was enough for AppKit to pick `.unifiedCompact` on its
own. With it, our column header's ink sits at y=60 against Finder's 61.

## Content blurs under the chrome: a scroll view must FILL ITS PANE (solved 2026-09-12)

THE SCROLL EDGE EFFECT IS NOT ABOUT THE TOOLBAR, IT IS ABOUT THE SCROLL VIEW'S FRAME. On Tahoe a
scroll view gets the effect only when it fills its pane top to bottom. Anything stacked as a SIBLING
above it - a column header, a filter chip - takes it out of contact with the safe area, and the
effect silently does not apply: content then clips at a hard line instead of blurring under the
chrome. This is why `scrollEdgeEffectStyle(.soft, for: .top)` appeared to do nothing however high or
low it was applied. The modifier was never the problem.

THE FIX IS `safeAreaBar(edge: .top)` (macOS 26). The bar sits in the safe area and the scroll
content passes under it, which is exactly what a column header is. All three views now use it - the
folder browser's header, the Photos browser's header, and the detail pane's file-query/qualifier
chip - and content visibly ghosts under them and under the toolbar.

How this was found, since three earlier guesses were wrong: Sarah Reichelt hit the AppKit twin
(troz.net, Jan 2026) - an `NSTableView` whose rows scrolled INTO the header with no blur, cured by
letting the enclosing `NSScrollView` span the whole content view. An Apple forums thread on sticky
section headers names `safeAreaBar` as the API for a pinned header that the edge effect respects.
Jon Sterling's Mastodon thread (Aug 2025) has a second, unrelated AppKit trap worth knowing: the
effect is also not applied when a scroll view has no vertical scroller, so `hasVerticalScroller =
true` matters even when the scroller is meant to be hidden.

What was tried and did NOT work, so it is not tried again: `scrollEdgeEffectStyle` on the list, on
the browser's root and on the whole detail pane; `NSSplitViewItem.titlebarSeparatorStyle` (a SwiftUI
window has no `NSSplitViewController` to set it on); and `titlebarAppearsTransparent = true`, which
does let content through and takes the toolbar's backdrop with it, so rows run across the buttons.

ONE THING STILL NOT MATCHED:

- THE SIDEBAR SELECTION IS STILL SWIFTUI'S, not Finder's. Finder fills a selected source-list row
  with light grey and tints its icon and label with the accent colour. `.tint` on the List does NOT
  drive that fill on macOS - it recoloured only the labels, giving blue on blue. Matching Finder
  here means hand-drawing the row background and giving up native focus and keyboard selection,
  which is the trade this note has refused twice; System Settings, also SwiftUI, draws the same
  solid pill we do.

ONE FILLED-TOGGLE STYLE (`ToolbarToggle`, App/Design.swift). The OCR toggle used to hand-draw its
on state - an accent `Circle` behind a white glyph, painted inside the label - which had to be
re-derived for the next toggle that came along and was a guess at the platform's look rather than
the platform's look. `Toggle` + `.toggleStyle(.button)` is the control the system provides: the
item's own capsule fills with the accent colour and the glyph inverts, which is what Finder does,
and it stays correct through theme, accent and appearance changes for free.

SERVE OVER HTTP SITS WITH THE MODE TOGGLES, not with Share. It is the same KIND of control as the
sidebar and OCR toggles - a global mode of the app, on until turned off - rather than an action on
whatever is on screen, so it shares their capsule and is present in every mode. Same switch as
Settings > Serving, never a second source of truth.

SEARCH BY A FILE IS A TOOLBAR BUTTON, not a glyph inside the search field. It was in the field, and
that had two problems: as an affordance it was invisible, and it had to hide itself whenever the
field held text - so it vanished exactly when a query was on screen. It now sits with a Share
button immediately before the field, in the same order and the same slot OCR mode puts its own
`Open Document` / `Share` pair, and wears the same `folder` symbol as OCR's. All the in-field
accessory machinery (`installAccessory`, the tag, the shrunk magnifier) is gone.

For Share to mean anything while browsing, BOTH BROWSERS NOW PUBLISH THEIR SELECTION to the model
(`selectSingle`) as well as keeping their own. The model's selection is what every selection-driven
action already reads - toolbar Share, the File menu, Quick Look - so a browsed file used to be
clickable but not actionable from anywhere outside the browser view.

ONE CONTEXT MENU FOR A FOLDER TOO (`FolderMenuItems`, App/FileMenu.swift). The sidebar's roots and
the browser's subfolders had drifted badly: the browser's menu was iconed and led with Open, the
sidebar's was icon-less, led with Pause, and offered no way to open, scope a search to, or copy the
path of the folder you had just right-clicked. One builder now serves both, root actions included.

Two things that shape it:
- "Open" and "Search in this folder" were literally the same call under two labels. They are now
  genuinely different: both enter the folder, the second also puts the caret in the search field
  (`SearchFieldFocus`, shared with the Find command and with the window's own focus-on-appear).
- "Remove from Omni" means one thing to a reader - Omni stops covering this folder - and needs two
  mechanisms. A ROOT is un-added. A SUBFOLDER has no such record, so the equivalent is an
  `.omniignore` rule via `ignoreFolder`, which also prunes what is indexed under it and is
  revertible in Settings > Content.
- PAUSE IS ROOT-SCOPED IN THE ENGINE and is shown only on a root. `pausedRoots` is consulted when
  roots are collected into a pass; nothing tests a crawled path against it, so offering it on a
  subfolder would set a flag that changes nothing.

Splitting the sidebar was forced, not chosen: with the folder row, its four badge states and this
menu inline, the `List` literal stopped type-checking in reasonable time. `FolderRow` and
`PhotoSourceRow` are now their own views and the badge predicates live in `RootIndexState`.

ONE CONTEXT MENU FOR A FILE (App/FileMenu.swift). The folder and Photos browsers used to carry two
items - Open, Reveal - against the results' twelve, so the same file offered less depending on which
list you reached it from. `FileMenuItems` is now the single builder for all three: Open, Quick Look,
[passages], Find similar, Generate Tags, Reveal, Copy path, Share, Move to Trash, [Select all],
Ignore folder. What stays with its own list is what only that list has - stacks and matching
passages in the results (through the `passages` slot, so the results menu keeps its exact order),
multi-selection actions (the browsers select one row), and the folder menu's Open / Search in this
folder / Visualize, since folders never appear in search results.

Two conditionals inside it are deliberate, not oversights: Generate Tags appears only for media
(`taggableKinds`), because a text file's snippet is a real excerpt and tags would be a downgrade;
Move to Trash is absent for a Photos asset, because deleting one deletes it from the library and
every synced device - that is Photos.app's offer to make, not Omni's.

## Real-time progress in the folder browser (2026-09-12)

Two things the browser now does while an index pass is running: the Files Indexed counts climb
without navigating away and back, and a folder that is filling carries the sidebar's pie at the
trailing edge of its Name column (a badge on the icon, in gallery view).

THE REFRESH IS PACED BY WHAT THE QUERY COSTS. `indexedChildrenDetailed` walks the whole subtree of
the browsed folder on the store's SERIAL queue - the same queue the indexer writes on. Measured
idle (`OMNI_PERF_LOG=1`, the `browse-list` line): 0.7 ms on a 105-file folder, 66 ms on 65k, 417 ms
on 23k, and 2.4 SECONDS on a 2.4M-file root; under load the same query on ~/Documents went from
68 ms to 648 ms. A fixed 2 s poll would have spent most of a big root's wall clock re-listing it.
Each round therefore sleeps 20x the time the last one took (floor 1.5 s, ceiling 30 s), holding the
duty cycle near 5% whatever the folder, and self-correcting as the store gets busier. It only runs
while `isIndexingUnder(folder:)` - an idle window costs nothing.

THE RING'S SIGNAL IS THE COUNT GOING UP, and getting there took three tries:
- every descendant of a working root: 25 identical empty rings down a listing, none of them moving.
  Says only what the sidebar already says.
- `progress.currentPath`: too narrow to ever fire. A pass interleaves roots and moves through a
  tree far faster than anyone can open a folder; instrumented over several minutes the current file
  never once landed inside the folder being browsed.
- a child whose indexed count GREW since the last listing. That is the same number the user is
  watching, so the ring appears on exactly the rows whose figures are climbing. Held for 45 s,
  because consecutive listings can be tens of seconds apart on a large folder and a ring that
  blinked out between them reads as stopped.

A SUBFOLDER BORROWS THE ROOT'S WEDGE. It has no clock of its own and one cannot be invented: the
index knows how many files it HAS under a folder, never how many it is going to get, so a
per-folder percentage would be a made-up denominator. The pass covering it does have a real clock -
the same one the sidebar draws - so the wedge is that, and the tooltip names the root rather than
implying the folder itself is that far along. An empty ring that never fills is not progress.

Three bugs found on the way, all of them invisible without the instrumentation:
- the CATCH-UP pass (the one that runs when you add a folder) merges progress field by field and
  never carried `currentPath`, so anything keyed off it was dead during exactly the pass that
  matters. The full pass assigns the whole struct.
- `currentPath` was set by the CRAWL only, which races far ahead of the encoder and on a small root
  finishes in a blink. It is now also set where chunks are WRITTEN - which is what "being indexed
  right now" means, and what Settings' caption wanted all along.
- the growth baseline survived a navigation, so every row of a folder you had just opened looked
  like a brand-new one that had grown: a listing full of rings where nothing was happening.

THE PIES ARE SAMPLED INTO `@State`, TWICE A SECOND, never read from the model inside a row.
`AppModel.progress` is one observable property that changes on every indexed file - hundreds a
second - so a row that reads it re-renders the whole listing at that rate. The sampled dict is only
assigned when it actually changed, so an idle listing re-renders not at all.

BACK/FORWARD STRESS: `Scripts/navstress.py <seed> <n> <x> <y> <w> <h>` - Cmd-[ / Cmd-], clicks
on either half of the nav capsule, double-clicks that descend, and bursts of 8 alternating. 200
actions at seeds 1337 and 4242: no crash, no new .ips, app still answering Accessibility queries at
0.9% CPU, title and chevron enablement still correct afterwards. One thing to know: a "descend"
double-click that lands on a FILE opens it in another app, which then takes the front window - keep
the click column left of any other window on screen.

Two traps met while building it:
- the `files.kind` COLUMN is an interned id, not a `FileKind` ordinal. Decoding it directly is a
  silent mis-mapping. `fileStatus(paths:)` already returns the kind as a String, along with size,
  mtime and `indexed_at` - build on that rather than on raw SQL.
- sort lives in the BROWSER, not in the toolbar's Sort menu. That menu governs search results,
  which are ranked by relevance and have no header to click.

## Browsing a Photos source (App/PhotoSourceBrowser.swift, 2026-09-12)

Selecting a Photos source used to do NOTHING: `Sidebar.onChange(of: selection)` handled `.folder`
and let `.photos` fall into the else, so the row took the highlight and the pane never moved.

It is NOT the folder browser with a different root, and cannot be. A photo is indexed at
`photos://all/<ASSET-UUID>%2FL0%2F001/<name>`, so every asset is its own ONE-FILE directory -
`indexedChildren(ofFolder: "photos://all")` returns no files and one UUID-named folder per photo.
A library is a flat gallery, so this lists the assets and draws them.

Two things that were tried and do not work, so they are not worth retrying:
- `filterFolder` cannot carry the source. It is a `URL`, and `URL(string: "photos://all")?.path`
  is empty - `all` is a HOST, not a path. Hence `AppModel.browsedPhotoSource`. The store side is
  fine: `folderPrefix` is a plain string prefix test, so the key works there directly.
- `ResultsList(results:)` cannot render it. It takes a `results` array but iterates `model.groups`
  to DRAW, so it is wired to the live search result set and silently ignores what it is passed -
  it rendered a blank pane. Browsing is not a search (no query, no ranking), so the grid and list
  are drawn in the browser, in the same shape `FolderBrowser` uses.

The listing is `store.listMatching(filter:topK:)` - a query-less lister that already orders by
mtime with path as the deterministic tiebreak. Capped at 1000, and the header says "first 1,000"
rather than quietly showing a prefix.

SUPERSEDED: the ghost rows below were then FIXED - see "Emptying a Photos library left its rows
immortal" in index.md. Kept for how the three causes were told apart.

THE THUMBNAILS ARE GHOSTS, AND IT IS A REAL BUG - just not in this view. They render as
file-type placeholders, and the cause was measured, not guessed (`OMNI_PERF_LOG=1`, the
`photo-thumb` line in `Thumbnail.load`):

    photo-thumb nil auth=2 assetFound=false id=23596587-...  (12 of 12)

`auth=2` is authorized, so it is not TCC - and the bundle id is `io.hanxiao.omni` either way, so
a `.build` copy shares the grant with the installed app. `assetFound=false` means
`PHAsset.fetchAssets(withLocalIdentifiers:)` returns NOTHING: the index holds 12 photo rows whose
assets are gone from the library. Not iCloud either - `PhotoLibrary.image` already falls back to
`.fastFormat` for the Optimize-Mac-Storage case (issue #13), and that fallback is never reached
because there is no asset to ask.

So the index is not reconciling deleted Photos assets the way it reconciles deleted files. The
stale sweep in `Indexer` is filesystem-shaped - it deletes `known` paths not in `seen`, gated on
`underPassRoots` - and a `photos://` root only gets swept when that source is actually crawled in
the pass. Ghost rows are worse than they look: they are unsearchable in practice, since a hit on
one shows a placeholder that cannot be opened.

NOT FIXED HERE ON PURPOSE. The fix is in deletion logic, which is the dangerous kind to change
speculatively - a wrong `underPassRoots` predicate deletes live rows. `PhotoLibrary.debugAssetExists`
and the gated `photo-thumb` log exist so the next person can tell the three causes apart in one
run instead of guessing between TCC, iCloud and staleness the way this took.

## Folder map, per folder (2026-09-12)

The map was never root-only in the ENGINE: `VectorStore.vectorsUnderFolder` takes a path PREFIX,
so it has always been recursive and has always worked for a subfolder. All that was missing was a
way to ask for one. `AppModel.visualizeFolder(_:umap:)` is that way, and the sidebar's two flat
map items plus the folder browser's subfolder rows (list AND grid - one `menu(_:)` builder serves
both) now route through it as a "Visualize" submenu of UMAP / PCA.

Two orderings inside it are load-bearing, and both were learned the hard way:
- `filterFolder` is cleared FIRST. Browsing wins the empty-result region (`showsFolderBrowser` is
  checked before `showsFolderViz`), so a map requested while a folder is being browsed would fit
  and then never be seen.
- the folder is pointed at BEFORE the mode is flipped, because `mapUsesUMAP.didSet` refits
  `selectedFolderForViz` synchronously. Setting the mode first fits the folder you are LEAVING and
  throws it away a moment later.
- it deliberately does NOT touch the sidebar selection. Selecting a folder means BROWSE, and the
  selection change calls `enterFolder` straight back over the map.

Release timings, 65,879 files under ~/Documents, `OMNI_PERF_LOG=1`:
UMAP pull 367 ms + fit 1075 ms; PCA pull 209 ms + fit 91 ms. The same operations in a DEBUG build
take tens of seconds - do not judge map performance from a debug run, and do not "optimise" it on
that evidence.

## Settings controls that were not native (2026-09-11)

- File-type rows carried a standing `line.3.horizontal` grip. That is an iOS edit-mode idiom;
  macOS reorders rows by dragging them with nothing drawn. The grip was never the handle either -
  `.draggable`/`.dropDestination` are on the WHOLE row, and the glyph was a plain `Image` with no
  gesture - so removing it cannot change the behaviour. NOTE: drag-to-reorder could not be
  exercised from the harness; synthetic `CGEvent` drags do not start an `NSDraggingSession`, so
  `.draggable` never fires. It needs a real mouse to confirm, and it is unverified either way.
- Those same rows used `.controlSize(.mini)` switches while "Generate tags" two sections down used
  the default size. Two switch sizes in one window. Now both default.
- `Toggle("", isOn:)` in those rows left VoiceOver reading an unnamed switch. Titled, then
  `.labelsHidden()`.
- "78 / 105" as a folder's progress reads as a ratio, and sat next to rows saying "65,876 files".
  macOS spells it "78 of 105". Three sites.
- The memory slider's two end labels were in different units: the word "Off" against a bare
  "128", under a value reading "6 GB". Now "128 GB".
- The Serving bearer token row had NO LABEL - "Bearer token" was the TextField's placeholder, so
  the row went blank-but-for-two-buttons the moment a token existed. It is `LabeledContent` now.
  Getting that right took three passes and both states have to be checked: a fixed 232pt field
  clipped the last two characters of a real 32-char token, and dropping the prompt left an empty
  token as two buttons with nothing between them (a borderless field in a grouped form is
  invisible until it has text). `maxWidth: .infinity` plus `.multilineTextAlignment(.trailing)`
  plus `prompt: Text("Not set")` handles both. To re-test, set `omni.serving.token` in defaults
  with the app CLOSED and put it back afterwards - do not type into the live field, every edit
  restarts the server.

Checked and left alone: the "Skip files smaller than" numeric fields (bordered, right-aligned,
units aligned in their own column - that is what a grouped form does), the composition bars with
legends in Storage and Performance, and the monospaced code blocks in Content/OCR/Serving.

## Compared against Finder, side by side (2026-09-11)

Both windows on screen at the same size, same folder, same view mode, screenshots diffed. What
follows is the result, so the comparison need not be redone.

Already matching, do not "fix":
- Toolbar grouping. Finder groups per FUNCTION into separate glass capsules - view modes in one,
  group-by in its own, the three actions in a third, search separate. Ours is the same shape:
  sidebar+OCR leading, then star / filter / sort / view as their own capsules. Finder does not put
  its sort control in with its view control either.
- Back/forward leading with Cmd-[ / Cmd-], flexible spacer, trailing cluster. Same SF Symbols as
  Finder throughout (`square.grid.2x2`, `list.bullet`, `sidebar.leading`, `chevron.backward`).
- The toolbar/content boundary. SUPERSEDED - this claimed Finder clips at a hard line too. It does
  not; both its views use the soft scroll pocket. See "Folder browser columns".
- SIDEBAR SELECTION: see "Two things not matched" under the toolbar audit - raised again and
  re-tested in 2026-09-12, same conclusion, with the `.tint` route now ruled out too.
- SIDEBAR SELECTION IS NOT A BUG. Finder draws a subtle fill with an accent-tinted label; we draw
  a solid accent pill with white text. Checked against System Settings, which is SwiftUI like us:
  it draws the solid pill too. Finder differs because it is an AppKit `NSOutlineView` in
  source-list style. Ours is a plain `List(selection:)` + `.listStyle(.sidebar)` - already the
  correct native rendering for the framework. Matching Finder here means hand-drawing row
  backgrounds and giving up native focus and keyboard selection. Don't.

Closed in this pass:
- View segments now run gallery-then-list, Finder's order (icon view first). They used to run the
  other way, which contradicted Finder AND this app's own Cmd-1 gallery / Cmd-2 list.
- The folder browser's list got `.alternatingRowBackgrounds()`. Finder's list view separates rows
  by striping them, not by drawing rules - which is the same reason the separators here are
  hidden. Finder stripes the empty area below the last row too, and so do we now.
- Date Modified and Size columns in the folder browser, to Finder's alignment: date reads from the
  left of its column, size from the right. Sizes go through `ByteCountFormatter`, not
  `Int.formatted(.byteCount(style: .file))` - the modern API renders SI case ("454 kB") where
  Finder writes "454 KB". `.fileSizeKey` rides along in the existing prefetch, so it costs no
  extra stat.

Known, deliberate differences that remain:
- SUPERSEDED: No Kind column and no clickable column-header row. Both exist now - see "Folder
  browser columns".
- SUPERSEDED (the chevron was removed, see "Folder browsing"): Folder rows carry a trailing
  `chevron.right`, not a leading disclosure triangle. Finder's LIST
  view expands in place; this browser DESCENDS, which is Finder's COLUMN view idiom, and the right
  chevron is the honest symbol for it.
- The search results list is not a Finder list and should not become one - it carries a snippet
  and a score, which is the product.

## Markdown panes (App/MarkdownBlock.swift, MarkdownSource.swift)
- LaTeX rendering was built and then REMOVED on purpose. This is a transcription workspace, not a
  rich editor: the model emits usable LaTeX, but setting it properly needs either a dependency or
  a maintained subset renderer, and neither earns its keep here. Do not re-add it without a reason
  beyond "the model emits it".
- Packages surveyed 2026-09-09 and rejected, so the search need not be repeated:
  `krzyzanowskim/STTextView` is GPL v3 or a paid commercial licence - an Apache-2.0 notarised app
  cannot take it, whatever its line-number gutter is worth. `CodeEditSourceEditor` (MIT) says of
  itself it is not production ready and drags tree-sitter plus grammars in. `gonzalezreal/Textual`
  (MIT) is the right shape - Markdown, maths, highlighting, selection in one - but needs macOS 15
  and we ship 14. `swift-markdown-ui` is in MAINTENANCE MODE, superseded by Textual.
  `gonzalezreal/swiftui-math` fits on paper (MIT, macOS 14, no runtime deps) and rendered CLIPPED
  fragments in this layout through two integration attempts. `swiftlang/swift-markdown`
  (Apache-2.0, Apple) is the parser to reach for if the hand-rolled block splitter ever needs
  replacing.
- The find bar exists because a searchable PROMPT only shows while the field is empty, so a match
  count put there vanishes exactly when there is one.
- The transcript panes are a `LazyVStack` of page sections, and each `ForEach` element resolves to
  exactly ONE subview - a `VStack` holding the page break and the section together. Those two facts
  are one fact. Laziness was tried early, broke `scrollTo` (following a run stuck a page behind) and
  left the pane blank for whole seconds, and was reverted to an eager `VStack` as "laziness cost
  three bugs and bought nothing". That diagnosis was wrong. WWDC26 session 321 states it plainly: a
  lazy stack addresses its subviews BY INDEX, and programmatic scrolling wants each `ForEach`
  element to resolve to a single subview. The old body emitted `PageBreak()` AND the section - two
  subviews for every element but the first - so the indexing the lazy stack relies on never matched
  the ids `scrollTo` was given. Wrapping the pair restores the mapping, and with it: clicking a
  distant thumbnail lands on that page, a live run follows correctly, and the eager cost is gone.
  Measured on the 40-page transcript, entering split from raw: 2513 ms eager, 164 ms lazy.
  Do not re-flatten that `VStack` for tidiness - it is the whole fix.
- `.defaultScrollAnchor(.bottom)` does NOT follow growing content here - measured, the view stayed
  on page 1 for a whole run. The tail is followed by scrolling to a zero-height anchor placed after
  the sections, which doubles as the clearance under the floating readout.
- SwiftUI text selection does not span sibling `Text` views, so one view per Markdown block means a
  drag stops at every paragraph. Finished sections merge their contiguous prose into a single
  `AttributedString`; a section still decoding stays per-block, because the arrival fade is driven
  by insertion and text appended inside one string cannot animate.
- The hand-rolled LINE-NUMBER GUTTER is gone. A `Text` per line in an eager `VStack` beside the
  editor left the window's SIDEBAR drawing nothing and unscrollable on any real transcript - the
  page navigator went blank the moment a run finished, which is how it was found. It also numbered
  logical lines, which a soft-wrapping editor does not lay out one to a row, and `Text("\(line)")`
  is a LocalizedStringKey, so past 999 it rendered "1,300" into a column sized for four digits.
- A page that is RUNNING but still empty is not a section. With one page in flight it never showed;
  with a group of 32 it put a wall of bare page rules on screen, one per page in the group, and the
  text arrived under the reader's scroll position. `sectionIDs` admits a running page only once it
  has text.
- While a group is in flight the transcript follows `visibleIndex` (the first page of the group,
  which is what the navigator marks), not the tail: every page in the group grows at once, so the
  tail is the LAST page of the group and following it left the rail and the text on different pages.
- SPLIT AND TRIPLE ARE ONE `switch` CASE, and that is a performance fix, not tidiness. As separate
  branches SwiftUI saw two unrelated hierarchies and rebuilt all 40 sections in BOTH text panes on
  every toggle: measured at 2693 ms on the 40-page transcript, against 13-54 ms once the columns
  keep their identity and only the image column comes and goes.
- Where that 2.7 s actually was, measured rather than guessed: `MarkdownBlock.parse` 16-23 ms and
  `MarkdownSource.highlighted` 0 ms once warm (it was already memoised; `parse` now is too). The
  rest was SwiftUI constructing and laying out the eager 40-section stacks themselves, which is
  what the lazy fix above removed: raw into split went 2513 ms -> 164 ms. `PageImage` was the
  obvious suspect and is innocent: its decode is 0.4 ms. Keeping hidden panes mounted was the
  other candidate and is NOT needed - it would have traded a switch stall for a relayout on every
  geometry change.
- The readout is PAGES FINISHED over pages queued: "6 of 24 pages". It used to name the page being
  worked on, which a group made meaningless - there is no single page - and naming the group's
  range instead ("Pages 1-32 of 40") answered the width of the batch when the question is how far
  along the document is. An earlier rule counted within a drop batch to stop a mid-run drop
  renumbering "page 13 of 40" into "page 13 of 41"; a finished COUNT does not have that problem,
  since a new file moves the total and leaves the count alone.
- Thumbnails cast a shadow and are NOT stroked. Preview's pages read as paper because of the
  shadow; a hairline on top of it is the frame a page rail is not supposed to have.
- The sidebar toggle is OURS, not the system's. The automatic one lives in the sidebar's own
  toolbar section and goes away with it, so folding the drawer left no way to unfold it but the
  View menu. `NavigationSplitView(columnVisibility:)` plus `.toolbar(removing: .sidebarToggle)` on
  the sidebar content, and a button in `.navigation`.
- SUPERSEDED (search by file is a toolbar button now, absent in OCR mode per the toolbar table):
  The in-field search-by-file button is hidden in OCR mode: the field finds text inside the open
  transcript there, and picking a file to search by is not something it can do.
- The drawer is narrower in OCR mode, and `navigationSplitViewColumnWidth` alone CANNOT do it: the
  window restores the divider it was last left at and that restoration wins, so a declared width
  only took effect when the mode was toggled and a cold launch straight into OCR opened the rail at
  the search sidebar's width. Declaring the range is still right, but the width is corrected in
  AppKit (`WindowTitleHider`'s tuner sets the divider once per wanted value - never on every window
  update, which would fight a drag). The leading toolbar items move with the divider, which is the
  price of the narrower drawer, not the inspector bug that moved them 300pt.
- SUPERSEDED by `ToolbarToggle` (see "Content blurs under the chrome"): The OCR toggle's "on" fill
  is drawn INSIDE its label, not with `.borderedProminent`: a prominent
  button takes a background of its own and broke the item out of the glass capsule it shares with
  the sidebar toggle, so OCR mode had two separate toolbar surfaces where every other mode had one.
- Leaving OCR mode used to block the main thread for 82 ms on a 40-page transcript: `deactivate`
  released 4.5 GB of weights and `setOCRResident(false)` called `MLX.GPU.clearCache()`, both
  between the click and the next frame. Both now happen on a utility queue and the measurement is
  0 ms. Returning costs ~19 ms, which is the editor rebuilding its text storage from the whole
  document (54 ms the first time, before `MarkdownSource`'s cache is warm) - one frame, left alone.
- Side-by-side keeps its halves together by SECTION AND FRACTION, and three things had to be true
  before any of it worked. A coordinate space named ON a `ScrollView` is the CONTENT's, so the
  probes reported once at layout and never again - it has to be named on a view outside the
  scroller. The lead is claimed by whichever half MOVED (the follower stays quiet for 200 ms after
  being scrolled); hover and `onScrollPhaseChange` were both tried and neither is the same
  question. And the raw pane is the SECTIONED one in a comparison view, never the editor: an
  `NSTextView` has no sections to report from or scroll to. Alignment is page-level - the same page
  sets two to four times taller as source than as prose, so an exact match is not available.
- A run STOPPED leaves its untouched pages `.stopped`, not `.pending`. They stay in the rail,
  dimmed, and clicking one transcribes it; the run loop takes the next `.pending` page wherever it
  is rather than walking a cursor, so a re-queued page behind the last one done is picked up. Before
  this, dropping a new file quietly resumed the document someone had just stopped and did the new
  one after it.
- The page navigator is an EAGER `ScrollView`, not a `List`. A sidebar list draws its own row
  chrome - the system selection, which greys out the moment the text pane takes focus, and a hover
  fill on top of it - so the accent mark this rail needs sat inside a second background.
  `.listRowBackground(Color.clear)` does not remove either. The only thing the List was still
  buying was scrolling to a row it had not built, and an eager stack does not have that problem.
  Selection means "the page you are looking at", not "the focused row", and is drawn once, by the
  thumbnail: the accent fill around the page AND its number, as Preview does.
- The editable source pane is an `NSTextView`, not `TextEditor`: the navigator has to scroll it to
  a page, which needs a character offset and a scroll call `TextEditor` does not expose. The
  offsets are built WITH the string (`documentSource()`) rather than found in it, because a page's
  own content can contain a `---`. Scroll with the page's WHOLE range - `scrollRangeToVisible`
  moves the least it can, so a one-character range below the fold lands at the bottom edge - and
  not by asking the layout manager for a rectangle, which is TextKit 2 here and has no geometry for
  a range it has not laid out.
- `FindHighlight.mark` and `MarkdownSource` set every colour in BOTH the SwiftUI and AppKit
  attribute scopes. `Text` reads
  the first; the editable pane's text storage comes from the second, and setting only one left the
  source view black.
- A DOCUMENT'S ID IS NOT ITS INDEX, and conflating them broke the tab bar twice over. Ids were
  `documents.count + n`, so closing a tab and dropping another handed the new one an id a
  surviving tab still held - and a `ForEach` keyed on a duplicate id renders and hit-tests the
  wrong row. On top of that the bar passed `doc.id` to `selectDocument`/`closeDocument`/
  `progress(ofDocument:)`, which all INDEXED `documents` with it, so once a close made the two
  diverge, clicking one tab brought up another. Ids now come from a monotonic counter that is
  never reused, the selection is held as `selectedDocumentID`, and every lookup is by id. The
  rename is the guard: an index assigned to it no longer compiles.
- There is NO native macOS control for document tabs inside a pane. `TabView` on macOS is a
  segmented control, and real document tabs are `NSWindowTabGroup`, a WINDOW feature. The custom
  bar is the right call; when it misbehaves the bug is in the model, not the control.
- The tab bar uses `windowBackgroundColor` for the track and `controlColor` for the selected
  capsule. Two shades of `.background` left them the same colour in light mode, so nothing read as
  raised; the semantic pair keeps the capsule lighter than the track in both appearances.
- A drop ADDS a tab. The decode queue lives on the session and the run loop reads it by index, so a
  file opened mid-run extends the queue instead of interrupting it; the new tab comes forward and
  is pinned, so the run behind it does not pull the view back. Document edits are keyed by tab for
  the same reason - one buffer showed one tab's edit under another's name.
- Pause holds at a PAGE boundary, not mid-decode: holding mid-page pins the GPU working set with
  nothing to show for it, and the half-decoded page would have to be discarded or resumed from a
  partial transcript. The chip says "Finishing this page" until the loop actually reaches the hold.

## macOS 14/15 toolbar: the real cause (2026-09-22, verified in a macOS 15.7 VM)

The leading buttons (sidebar, OCR, serving) were present but COLLAPSED TO 10x10 on Sequoia.
68921d7 (move to `.primaryAction`) and db9c2a5 (titled labels) were reasoned from a dump without a
Sequoia machine and did not fix it: 0.13.6, which has both, still dumps 10x10 in the VM.
- CAUSE: the toolbar was REBUILT, and macOS 14/15 do not size SwiftUI items in a rebuilt toolbar.
  Two rebuild triggers: (1) the root was `Group { if ocrMode { split.searchable(A) } else if
  showsSearch { split.searchable(B) } else { split } }` - three split views, swapped on OCR toggle
  and on launch going ready; (2) `.toolbar` sat on a `Group` inside the detail, and a Group applies
  modifiers to each CHILD, so the toolbar belonged to the phase switch's content (FB13106004).
- FIX: one split view with one mode-aware `.searchable` (OCR binds it to find-in-document), and a
  `ZStack` owner for `.toolbar`. Items measure 34x28 / 31x28 / 29x28 at launch, through OCR on/off
  cycles and with results showing. Tahoe dumps identical to 0.13.6 in search and OCR mode.
- NO TITLE ON 14/15 EITHER (2026-09-23). The title item was what filled the free space; hiding it
  alone packed the search field against the leading buttons. `titleVisibility = .hidden` plus an
  empty `.principal` toolbar item fixes both: AppKit gives a centre item flexible room on each side.
  Checked on 15.7 in idle, results and OCR mode; the window keeps its name for the Window menu.
  The stretched-item recipe (low-priority 8000pt width) does fill it, but AppKit's overflow reads
  the request and pushes Search by File and Share into ». Trailing items are `.primaryAction` pre-26.
- The launch animation: `onChange(of: ocrDrawerWanted, initial: true)` set `columns` from
  `.automatic` inside `withAnimation`, animating the first layout (content slid in from the
  top-left). The initial call is now unanimated; a 60 fps recording shows no motion.

TESTING ON SEQUOIA: tart VM `sequoia` (macOS 15.7.7), TART_HOME=/Volumes/han2tb/tart/home, binary
/Volumes/han2tb/tart/tart.app/Contents/MacOS/tart, run with `--dir=share:/Volumes/han2tb/tart/share`,
SSH admin/admin. The shared folder serves STALE content for an overwritten path - copy every build
to a new directory name, or you test the previous build. `OMNI_UI_DEBUG=1` + `kill -USR2` dumps the
toolbar to /tmp/omni-debug-toolbar.txt inside the VM. The nano model runs in the VM (MLX works).

## Settings type system; one "weaker matches" button (2026-09-22)

- Settings fonts follow one rule, written at `SettingsView`'s root: row text body (values
  secondary), detail lines and footers caption secondary, code callout monospaced (the serving log
  small monospaced, backed by ~/Library/Logs/Omni/serving.log), paths never monospaced, no weight changes, no caption2, tabular digits set
  once at the root.
- "Show N weaker matches" is ONE button (`ContentView.weakerMatchesTitle`), bordered and large, in
  both the empty state and the results footer. The footer used to be a `.plain` label ("Show N more
  matches"); a plain button is hit only on its glyphs, so clicks between or beside the letters did
  nothing - the "sometimes not responding" report. Verified by clicking the button's edge.

## hanxiao.io/omni (redesigned 2026-09-22)

- `site/omni` is the page. `gh workflow run site.yml` deploys it WITHOUT an app release: it reads
  version and MD5 from the live latest.json, stamps the same three hooks a release stamps
  (`dl-ver`, every `href="Omni*.dmg"`, `dl-md5`) and purges Cloudflare for the page, faq.json and
  every asset. Keep those three hooks in any redesign; the release step rewrites them.
- Copy is lean: a headline and a few words per item, no explanatory paragraphs. FAQ answers live in
  faq.json and were fact-checked against the code; keep them true when features change.
- Screenshots (the gallery, 2026-10-13): NOT from a real index. Limiting `-omni.addedFolders` to
  three folders does not limit search - a clone of the real index still answered with agent session
  logs and a backup volume's files. The gallery is shot on a demo index of public content only: this
  repo, its SwiftPM checkouts, arXiv papers, the synthetic OCR bench scan, the films and their
  narration, in ~/Projects, ~/Papers and ~/Media (home paths so rows read `~/Papers`; none of them is
  a root of the real install), plus a second isolated instance ("Mac mini") for Linked Macs. Text
  first: one image-free query per card. Window 1440x900 (this display is 1x), captured with
  `screencapture -o -l <window>` so the window keeps its own corners, then a two-layer shadow baked
  into the alpha on a 1500x960 WebP. PerfScript `ocrview:` and `ocrpage:` set the OCR workspace's
  layout and page; synthetic clicks posted to the instance did not reach its toolbar. Check every OCR
  card against the scan: the handwritten pages misread digits (4.3 -> 7.8), the printed table pages
  are exact.

## Dragging files out (owner's decisions, 2026-10-02)

- A FILE DRAGGED OUT OF OMNI IS NEVER MOVED. The drag offers copy, link and generic, never move, so
  a drop in a Finder folder on the same disk copies (Option-Command still makes an alias). Omni is a
  view onto files that live elsewhere; a drag must not take one out of its folder.
- COMMAND-C ON SELECTED FILES PUTS THE FILES ON THE CLIPBOARD, Finder-style, together with their
  paths as text: Command-V in a Finder folder copies the files, Command-V in a text field pastes the
  paths as before. Through `OmniPasteboard.copyFiles`, like every copy Omni makes. Copy Path
  (Option-Command-C) stays text-only, as Finder's Copy as Pathname does.
- HOW (App/FileDrag.swift): an AppKit dragging session begun from a SwiftUI `DragGesture`'s first
  movement (`fileDragSource`), not `.draggable` - SwiftUI drags one item before macOS 26 and cannot
  restrict the operation. Finder's handles: a list row's icon and name, a grid cell's thumbnail and
  label; a drag that starts anywhere else still draws the marquee (the handle's gesture outranks
  the container's). Grabbing a selected item drags the selection in result order, an unselected one
  selects it and drags it alone (`AppModel.dragPaths`). Photos assets go as `NSFilePromiseProvider`s,
  exported only when dropped. Inside the app the operation mask is empty and both drop targets (the
  search area, the sidebar's add-a-folder) check `FileDrag.isActive`, so a result is never taken
  back in - the misclick that once made rows non-draggable.
- THE CLIPBOARD HISTORY NEVER SEES ANY OF IT, three ways: a drag travels on the drag pasteboard,
  which the history never polls; a copy carries a file URL on every item, which the history skips;
  and its first item carries `ownType` too. Checked by PerfScript `cliptest` on a PRIVATE named
  pasteboard (Omni's 3-file copy skipped, a Finder copy skipped, typed text recorded as the positive
  control; the real clipboard's changeCount did not move).
- THE TEST: `Scripts/drag-test.sh` runs `FileDragUITests` - real XCUITest mouse drags from a renamed
  copy of the app (never the real bundle id: launch() would quit the user's Omni) onto DropProbe
  (`Tools/dropprobe`, built by `Scripts/drop-probe.sh`), a floating window in another process that
  reports what arrived and which operations the SOURCE offered, in the accessibility value of
  `probe.report` (the runner is sandboxed; it reads the probe's UI, not a file). Six cases, ALL
  PASSING 2026-10-02 (228 s): a list row and a gallery cell each arrive as their file, with move not
  offered and the source still in place; a selected result drags the selection in result order;
  Omni refuses its own drag; a drag from a row's blank area is a marquee (3 rows selected) and lifts
  nothing; Command-C leaves file URLs + path text + the own marker (the user's clipboard is saved and
  restored around it). Needs an unlocked screen and automation mode.
- THE MARQUEE HAD BEEN DEAD SINCE ResultClick SHIPPED, 0.14.5 included, and this test is what found
  it. Every row carries a `DragGesture(minimumDistance: 0)` for its click, a child's gesture outranks
  a parent's plain `.gesture`, and rows fill the list - so the container's marquee never recognized,
  from anywhere. Shown with raw CGEvent drags and an accessibility readout of each row's selected
  state (positive control: a click reads as selected): no rectangle, nothing selected, in the release
  and in the build with the file-drag handles switched off. It is a `.simultaneousGesture` now, and
  `clickSlop` (16 pt) decides between click and band as it was written to; it stands down while a
  file drag is active (that one lifts at 4 pt).
- THREE TRAPS THAT COST A RUN EACH, so they are not paid again:
  - The test's files must live OUTSIDE the runner's sandbox container. The app under test writing
    into `~/Library/Containers/<runner>/...` is "data from other apps": its first `open()` blocks on
    a privacy prompt (main thread in `saveIgnoreText`), the app never becomes ready, and XCUITest
    reports "does not have a process ID" while the process sits there. The script makes the corpus
    and index in /private/tmp and passes the path as TEST_RUNNER_OMNI_DRAG_ROOT.
  - A SwiftUI `Text` reaches accessibility as a static text whose VALUE is the string; its
    description and title - what XCUITest calls `label` - are empty. Match `value`, not `label`.
  - A fresh copy opens in the GALLERY, whose cells are `result.item`, not `result.row`; set the view
    (Command-1/2). And near-identical corpus files stack as duplicates and leave too few rows.

## UI text and menus (2026-09-23)
- NO PROSE WHERE macOS HAS NONE. Open panels set no `message` (a verb `prompt` at most); alerts
  are a title plus one factual clause; launch screens, empty states and Settings footers do not
  explain internals or restate their control. Tooltips name the thing, not how to use it.
  The one exception is the Clipboard-off screen: it asks the user to turn a feature on, so it
  carries one line saying what that gets them (asked for on 2026-10-02).
- MENUS ARE TITLE CASE, context menus included ("Find Similar", "Copy Path", "Show in Finder").
  Reveal is "Show in Finder" / "Show in Photos" everywhere. An ellipsis only where a panel or
  dialog follows. Destructive items are the last group; one trash item per menu (a stack's
  "Move All N Copies to Trash" rides `FileMenuItems.trashAll`). Every Tahoe context-menu item has
  an icon; a checkable item is a `Toggle`, not a checkmark icon. `FileMenuItems` is THE file menu
  - results, both browsers and folder-map dots - so a file offers the same actions everywhere.
- CHECK THE LIVE MENU BAR, not the source: `osascript ... get name of every menu item of menu "X"
  of menu bar item "X" of menu bar 1` on the dev pid. It showed doubled separators that no code
  reading does - an inline `Picker` and a new `CommandGroup` bring their own.
- The About panel shows the mole without its tile (`Mole` in Assets.xcassets) and a link, no
  tagline.

## The mole glyph family (App/Assets.xcassets/Mole*, Tools/brand/mole_family.py)
One drawn mole, template-rendered so it takes `.tertiary` like an SF Symbol, with one gag per
screen: `MoleGlyph` (plain) onboarding, `MoleSerious` (determined brows) "Loading your index",
`MoleSearch` (star eyes) the search prompt, `MoleOCR` (reading glasses) the OCR drop zone,
`MoleSleep` (z z) "Add folders to search", `MoleEmpty` (x x eyes) "Nothing indexed here",
`MoleCurious` (one brow up, a question mark) "No recent items", `MoleSad` (worried brows, a tear
down each cheek, one falling) every no-results screen: "No results above N%", "No matches with
these filters" and "No matches" (they used SF Symbols, the only empty states that did not).
PNGs are rendered with `rsvg-convert -w 80/160/240` from the generator's SVG (matches the shipped
assets to 0.6/255 mean alpha difference). SAME SIZE AND COLOUR EVERYWHERE: every screen draws it
through `StatusGlyph` (68 pt, `.tertiary`); a one-off size or style on one screen is what made the
welcome mole read darker than the rest.
All are rendered from the full 100x100 canvas, never cropped, so the mole keeps one size and place
across screens. `StatusGlyph` draws any `symbol` starting with "Mole" as one of these. Change the
drawing in `mole_family.py`, not the PNGs.
`MoleHug` (2026-10-13, `mole_family.py hug`): two of the same mole cheek to cheek, happy eyes, one
mound and a heart - the "Linked with ..." sheet on both Macs. A first version had each hugging the
other; at 68 pt the two extra paws read as too many hands (Han), so they are gone. A 140x100 canvas at the family's scale (`rsvg-convert -w
112/224/336`), 68 pt tall, `.secondary` (a success, not an empty state); whatever is in front is
cut out of what is behind with the family's outline gap. Its buttons are a Tahoe alert's equal-width
pair, `.glass` / `.glassProminent` on 26, bordered before.
`MoleSearchLinked` / `MoleSearchLinkedMany` (`mole_family.py linked`): the search prompt while a
search reaches linked Macs - the search mole unchanged, with a half-size one leaning in on the left
for one linked Mac and one each side for two or more. Lean from the hips: body and head rotate, the
mound stays level. 176x100, rendered by HEIGHT (`rsvg-convert -h 80/160/240`) so the centre mole is
pixel for pixel the plain prompt's; `StatusGlyph` is now 68 pt tall with the width from the drawing.
The prompt counts their files too ("Search 2,482 files on 3 Macs"): online, sharing Macs only, from
the sources each reported on connect, and none while a folder on this Mac scopes the search.
The pairing code is six boxes in two groups of three (Liquid Glass on 26), as macOS shows and asks
for a verification code; on the typing Mac a clear field under the boxes takes the keys and a paste.

## History replay and the sidebar (2026-10-04)
- A HISTORY ROW REPLAYS EXACTLY WHAT WAS SAVED, whatever is set at the moment of the click. A text
  entry always did (its string carries every filter as a qualifier); a file entry (Find Similar,
  search by file) stored its filters beside the path and never read them back, so it replayed under
  the current scope: saved with 25 results in one folder, it came back with 34 from another. It now
  resets every filter and restores the recorded ones (`restoreRecordedFilters`).
- A FOLDER CLICK WITH A SEARCH ACTIVE KEEPS THE SEARCH AND REPLACES ITS SCOPE: a replayed
  `in:"A" type:text memory limit` becomes `memory limit type:text in:B`; a replayed file query moves
  to B. Verified through real sidebar selections (`sidebarhistory:<n>` in PerfScript) - calling
  `runHistoryQuery` directly leaves the sidebar selection where it was, and a re-selected folder
  then fires nothing, which reads as this rule being broken when it is not.
- SETTINGS: Clipboard moved to Files, with the other sources it is indexed beside. Folder ignore
  files are rows of folder name over location with a reveal arrow (actions in the context menu), not
  one middle-truncated path and a full-size button each. And only folders the crawl can enter: the
  watcher used to register an `.omniignore` anywhere under a root, including inside excluded
  folders such as `.build/`, where its rules can never apply.


## Sidebar: foldable sections, one row per day (2026-10-05)
- ORDER: Recents and Clipboard sit above every header, where Finder puts Recents; then Index,
  Bookmarks, History. Each section folds from the chevron the sidebar shows under the pointer
  (`Section(isExpanded:)`), and the folds persist in `omni.sidebarFolds` - never written by an
  isolated or UI-test run (`AppModel.persistsUIState`).
- HISTORY IS ONE ROW PER DAY with searches, replacing the Today / 7 Days / 30 Days ladder above
  (whose 23-day bucket held 68% of the list). A `calendar` icon, not a folder: folders in this
  sidebar are real ones. No count; searches indent 8 pt. Day names come from the system formatters
  (relative "Today"/"Yesterday", locale field order). Today and Yesterday start open, older days
  closed; a day opened or closed by hand stays that way (keyed by date, dropped after a year).
- DAYS ARE FLAT BUTTON ROWS, NOT `DisclosureGroup`: one disclosure triangle makes the List reserve a
  triangle column for EVERY row, which moved Recents and the folders 9 pt off Finder's edge. A
  Button rather than a tap gesture, so VoiceOver can open it.
- HEADER COLOUR: a sidebar List draws section titles at 171 grey on a light sidebar; Finder's
  measure 112, the secondary label colour (our row icons measure 107-112). `SidebarHeader` sets it.
  An unselectable row's text is dimmed the same way, so the day title sets `.primary`.
- Bookmarks draw with the search's own icon, not a star: the header already says it.
- TESTING TRAP: a synthetic click posted into a background instance toggles a Button but selects
  no List row - the first click in a non-key window only activates it. Row selection was checked
  through `sidebarhistory:<n>` instead.

## Update window release notes (issue #28, 2026-10-08)
- The changelog jumped between two positions when scrolled. `NSTextView(frame:)` is TextKit 2,
  which lays out only the visible text and estimates the rest; scrolling corrected the estimate,
  the document grew under the scroll position (432 -> 447 -> 462 pt over one scroll of three
  releases' notes, measured offscreen), and the text moved. TextKit 1 (`usingTextLayoutManager:
  false`) laid out in full before the alert shows holds one height (462) throughout. Any other
  scrolling NSTextView with more than a screen of text wants the same.

## Shortcuts window, folder names, history remove (2026-10-09)

- THE SHORTCUTS WINDOW IS READ OFF THE HANDLERS. It listed fifteen menu chords and none of the keys
  the content pane and search field handle themselves: Esc (clear search, stop transcribing),
  Return, Home/End and Option-arrows, Shift-arrows, type-select, Cmd-Up, Shift-Cmd-G,
  Ctrl-Cmd-1..9, Ctrl-Cmd-S and every transcript chord. Now four groups (Search, Results, Go and
  View, Transcribe) in two columns, 724x625 pt. A new chord goes into the list in the same change.
- FINDER'S NAME FOR A FOLDER. iCloud Drive is `com~apple~CloudDocs` on disk and the sidebar showed
  that. `SpecialFolder.name(for:)` returns `FileManager.displayName` for the special folders
  (iCloud Drive, and the localized Documents/Downloads/...), the last path component otherwise;
  used by the sidebar, Go menu, toolbar title, scope chips, history rows and Settings. Read per row,
  so a path whose leaf is not a special leaf never resolves symlinks.
- HISTORY ROWS REMOVE ON HOVER. The trailing glyph (kind or filter mark) gives way to an
  `xmark.circle.fill` under the pointer, the same command as Remove from History. A clear mark, not
  a trash can: nothing on disk is touched. Not exercised in a running app: synthetic mouse events
  posted to a background instance are dropped, and bringing it forward would take the user's focus.
- PerfScript `menu:<title>` performs a menu bar item by title, so a background instance can open
  the shortcuts window for `screencapture -l`.

## iCloud-only files in the lists (2026-10-09)

Checked through the app on a real iCloud folder after indexing with "Index, then remove download":
17 of 17 files stayed dataless through the folder gallery, folder list, results gallery and
results list - every thumbnail comes from the one iCloud keeps (QuickLook, no download).

- FINDER'S CLOUD BADGE (`CloudBadge`) after the name in all four views; white on a selected row.
  Lexical first: only a path under `Library/Mobile Documents` or `Library/CloudStorage` is stat'ed,
  off the main thread. Refreshed by Open and Quick Look (they download, as in Finder), by Download
  Now / Remove Download, and when the app comes forward (Finder may have changed one).
- DOWNLOAD NOW / REMOVE DOWNLOAD in the file menu under Open, iCloud Drive only. Measured: double
  click downloaded and opened in Preview in under 1 s, the badge cleared; Remove Download put it
  back in 0.5 s and the badge returned. Refused (beep) while Preview still had the file mapped,
  as iCloud refuses any eviction of a file in use.
- A SELECTION NEVER DOWNLOADS. `OpenAtHit.prefetch` read a selected PDF to find the hit's page,
  which for an iCloud-only PDF was a download on a click. It skips dataless files; the open does
  the work instead.
- THUMBNAILS COVER THE TILE (`Thumbnail.coverPixels`). Every path decoded to FIT the long side to
  the tile, and the tile FILLS a square, so a 1280x749 photo came back 256x150 and was stretched
  1.7x, a tall screenshot 2.8x: soft next to Finder, downloaded or not. ImageIO now sizes the long
  side from the header so the short side covers (capped at 4x); QuickLook is asked for a box twice
  the tile and the answer scaled down to the short side on QuickLook's queue. A 1:7.5 strip still
  shows a soft middle crop: iCloud's own thumbnail of it is 34x256.

## One action, one icon; plain text (2026-10-09)

- Every action that removes an entry from a list draws `trash`: Remove from History (row, day,
  hover button), Remove from Omni, Remove from Sidebar, Clear Clipboard History, Move to Trash.
  The hover button was `xmark.circle.fill` and Remove from Omni `minus.circle`: three icons for one
  verb. Remove Download keeps Finder's `xmark.icloud`; the toolbar's filled star is a state.
- Every copy draws `doc.on.doc` (Copy Page as Markdown was `doc.on.clipboard`, which is Paste's).
  The empty-state hints use the toolbar's and menus' icons: `folder` for Search by File,
  `sparkle.magnifyingglass` for Find Similar.
- UI text says what a control does, in plain words: no "the headroom comes back on its own", no
  "one batch of indexing always works in", no "every search you settle on". A new string is read
  next to its neighbours before it ships.

## One column header for every browser (2026-10-13)
- Four browsers drew four copies of the column header (folder, Recents, a Photos source, a linked
  Mac), and two had drifted far enough that Han saw them as different controls: the linked Mac's
  had an item count beside "Name", no sorting and sat as a sibling above the list (no Tahoe scroll
  edge); Photos' put its chevron on top of "Date Modified" and an item count beside "Name". The
  folder browser's was right. It is now `BrowserHeader` (BrowserColumns.swift) with
  `columnHeader { }` (the safe-area bar) and `reportsCellsMaxX` (the measured trailing pad), and all
  four use it, each passing its columns and its sort. Photos shows Kind, Date Modified, Size (the
  store's `listMatching` now fills `size`, which it had dropped); a linked Mac shows Kind, Date
  Modified, Size, Files Indexed from what its listing already carried. No item counts: Finder
  shows none in the header, and the folder browser never did.

## Search results in columns (Han, 2026-10-14)

The list view of search results was a three-line row of its own (40 pt thumbnail; name; snippet; a
metadata line of kind, size, position, folder and date). It now draws as the folder browser does:
one line, the shared `BrowserHeader` as a safe-area bar, `ColumnSlot` geometry and `BrowserMetrics`,
so a search and a folder are one kind of list. What is about the search comes first, right after
Name (Han: not pushed to the end): Relevance; Match, why it matched and where (Han's P2 item 16 -
"Frame at 4:00", "Video" for a video up to 4 minutes, which is embedded whole and so has no moment
(it said "Frame", which read as one particular frame; Han, 2026-10-15), "Speech at 4:00", "Words in
the image", "Page 100", "Line 1", "Scanned page 3",
"File name", "Contains INV-2041"; `SearchHit.matchReason`, with `SearchHit.via` from the store for
the name and identifier cases); Excerpt, the passage or a screenshot's read words; Tags, a picture's
or a frame's (Han: tags are not "Match"). Then the file's facts: Folder, Kind, Dimensions,
Duration, Size, the dates. The header's right-click menu chooses them in that order
(`BrowserColumnSettings.search`, kept apart from the folder browser's); the first view shows
Relevance, Match, Excerpt, Tags and Folder, with Name at least 180 pt. A header click sorts the
results (name, kind, folder, date modified, relevance) and a second click reverses it; the toolbar's
Sort menu sets the same order. The passages disclosure is the row's leading triangle, where Finder
draws one; an opened stack's copies are indented one icon width. The grid is unchanged.

The header cost +25 ms of main thread a search, and is now free. Measured in isolated instances on
one settled index, 15 x two searches in list view each, main-thread CPU per search (perf script),
arms interleaved: the build before columns 61-63 / 69-73 ms ("dog on the beach" / "invoice total"
p50), columns 84-89 / 97-99, grid equal in both (80 / 82-91). Switching parts off one at a time
put all of it on the header in its bar: without the Excerpt column, the folder tooltip, the rows'
width report, the header's own width measurement or its context menu, nothing changed; an
`.equatable()` header changed nothing; an empty bar, a bar holding one Text, or the header as a
plain sibling (no edge effect) each removed it. The safe-area bar sizes its content again whenever
the scroll view under it is replaced, which is every search (a new result set is a new scroll
view), and sizing the header's flexible columns there is the cost; the profile showed it as
window layout, with the toolbar laid out again (`NSToolbarView layout` 142 -> 309 ms over 30
searches). The bar now holds a clear spacer of the header's measured height with the header as an
overlay on it (`ColumnHeaderBar`): 58-65 / 67-74 ms over three reps, the pre-columns figure.
Screenshots of the search list and the folder browser are identical pixel for pixel before and
after, apart from the text cursor.


## Search rows against the sidebar (Han, 2026-10-15)

"This is too close to the left drawer", against Finder: the search list's rows began 2pt past the
floating sidebar's edge, disclosure triangles against it, 23pt left of the folder browser's rows. Not
a safe-area matter - the scroll view's frame starts at the pane edge (260) like the browser's list -
but an OVERFLOW: the default columns are 1,036pt (Name 180 + lead 38, Relevance 85, Match 181, Excerpt
160, Tags 197, Folder 177, trail 18) and a 1,280pt window's list has 988, so the row was wider than
the list and centred, 24pt under the sidebar and 24pt past the right edge (the header with it).
Rows and the shared header are now anchored leading (`frame(maxWidth: .infinity, alignment: .leading)`),
so extra width leaves by the trailing edge as in a too-narrow Finder list, and Excerpt's least width
is 0: the flexible column gives way first, and the default columns fit 1,280pt again. Measured at
1280x800: stripe 252 -> 276, icon 270 -> 294, name 291 -> 314 - the folder browser's 294 and 314.
A first try padded by the scroll view's leading safe-area inset and overshot: that inset reads 260,
the whole width to the window's edge, not the part the sidebar covers.

## Shortcuts, Siri, Spotlight and Services (2026-10-20, App/Intents)

- ONE SEARCH, ANOTHER DOOR. `Search Files by Meaning` (background, does not open the window) returns
  files with the passage that matched through `ServingController.searchForIntents` - the served
  search, its duplicate collapsing and its passages, the answer an agent gets over MCP - so a Shortcut
  can hand them to Use Model, Mail or Finder. `OmniFileQuery` is an `EntityStringQuery`: Siri passes
  words and the served search resolves them, which is the path Apple steers large, changing datasets
  to (rather than donating every chunk as an IndexedEntity, which would copy the index into the
  system's with a weaker model). `Search in Omni` and `Find Similar in Omni` open the window; the
  second, saved as a Quick Action, is a Finder context-menu item without a Finder extension.
  `OmniShortcuts` offers the first two to Spotlight and Siri with no setup.
- SERVICES > SEARCH IN OMNI for selected text in any app (Info.plist `NSServices`, handled by
  `AppDelegate.searchOmni`): the selection goes through `applyParsedQuery`, the door the box uses.
- NOT TESTED BY TRIGGERING THE SYSTEM: a Service or a Shortcut from outside would route to whichever
  Omni the system picks - the user's own, or a dev build launched on the real index. The build's
  `Metadata.appintents/extract.actionsdata` lists the 3 actions, the entity, its query and the 2
  App Shortcuts; `OMNI_PERF_SCRIPT="intent:<text>"` runs the action's own search in an isolated
  instance and logs it ("Kaution Wohnung": the Berlin lease's deposit line, then the English lease's
  security-deposit clause; 13-32 ms main-thread CPU a call).
- Intent searches land in the sidebar history marked as served (`ServedSurface.intent`).
- MACOS 27, NOT ADOPTED (Han, 2026-10-21: skip for now). WWDC26 gives an app with a large, changing
  dataset `.system.searchInApp` (opens the app's own search with the user's query - the documented
  path beside EntityStringQuery), `EntityCollection` (bulk parameters by id) and View Annotations
  (Siri sees the rows on screen); Core AI is Apple's new on-device inference framework, to be
  measured against MLX for the encoders. All four need the macOS 27 SDK, which only Xcode 27 has;
  this Mac builds with Xcode 26.6 on macOS 26.6.2 and release CI selects Xcode 26. When the toolchain
  moves, they go in behind availability checks, tested on a macOS 27 machine.
- RESULT ROWS SHOW THE PASSAGE THAT MATCHED (index.md, "Passages"), and an excerpt from inside the
  chunk starts with an ellipsis, as Mail and Spotlight show one: "...only after we found the cause:
  the raw water im..." at Page 41 for "boat engine repair", where the row showed the page's opening
  before. Open-at-the-match reads the snippet (words only, so the ellipsis is ignored) and now starts
  from the matched passage.

## A long book opened at its page; Settings within the screen (issues #26, #30, 2026-10-23)

#26, on a 4,273-page 288 MB PDF: a double-click opened Preview 20-30 s later at an unrelated page,
each further click opened it again, and the CPU stayed busy until quit. Three causes in OpenAtHit:
the phrase unique to the page was checked with PDFKit's findString, which on a mismatch reads on to
the end of the book (~2.4 ms a page, for up to hundreds of candidates, unbounded); the prefetch held
the page cache's lock for its whole 8 s budget, so the open waited behind it and then gave up and
opened plainly (Preview restored its last page); and every click queued its own open. Now the phrase
is built from runs of plain words within one line of the page, so it occurs verbatim in the page's
text and its uniqueness is a substring test - no search; the pages are read one lock-hold at a time
(PDFKit is still read by one thread at a time); one deadline bounds it all, and a long book's deep
page gets a ten-word phrase checked against the pages read so far; one open per file at a time. On
a generated 4,000-page book (a running header on every page): page 100 in 252 ms with every earlier
page checked, pages 2,000 and 3,990 in ~1.0 s; through the app, Return on the page-3,990 hit opened
Preview at "Page 3,990 of 4,000" with the phrase found and highlighted (1,118 ms). The reporter's
other question - only the first 2 MB of a PDF indexed? - was the Settings text, which listed text,
Markdown, web pages and code as read in full; PDFs and converted documents are read in turns to the
end (that book's facts on pages 100, 2,000 and 3,990 found at those pages), and it says so now.

#30: Settings sized every pane to its full height (`fixedSize`), and Performance is ~1,150 pt
against a 13-inch MacBook's ~900. The window is capped at the screen's usable height less its
chrome, and a taller pane scrolls inside it, as System Settings does (`-omni.settingsScreenHeight`
stands in for a smaller screen in a screenshot: 848 pt on a 900 pt screen).

## Preview runs the search; a passage opens at its own words; the passage rows (#26, #31, #32, 2026-10-24)

#26, the reporter on 0.19.2: Preview came up at its last page with the passage's phrase typed into its
search field and not searched, until a click on the page. Reproduced on a generated 4,000-page book:
opened fresh, the phrase is found (Page 3,990 of 4,000); a second open with another phrase 6 or 15 s
later, the book already open, fills the field and leaves the window where it was. The same open-
documents event sent again 0.4 s later runs it: from Preview not running, just opened and open for a
while, the book landed on the phrase's page every time (pages 3,990, 120, 2,500; a single send to the
just-opened book failed both times). So OpenAtHit sends a search twice.

#32: two rows under one file on the same page, double-clicked, searched the same words. A dense page
holds several chunks - each its own result, rightly - and the phrase was kept per page, so every
passage on it was sent the first one's. It is kept per passage now (the page and the four words the
row's chunk starts with), each built from where its chunk starts.

#31: the passage text faded at its top as well as its foot, and the first line was hard to read: the
foot only now (the locator says where it starts). A passage row highlights under the pointer, as it
opens on its own with a double-click, and shows a spinner in place of its score from the double-click
until its app has the file (a long book's phrase can take a second, Preview a few more). Not
reproduced: the sidebar losing its padding while typing, and a narrow search field at launch - not at
the reporter's 1,020 x 682 window, with a folder chip, history rows added while typing, or long ones
(perf script, isolated instance); asked for steps.

The reporter's questions, measured (isolated instance, Nano, GPU): a generated 3,500-page PDF with 34 MB
of text indexed in 178 s, every page in the index (16,614 passages, 3,500 distinct page locators), the
footprint 7.1 GB at its peak while indexing, 2.75 GB at rest, 3.06 GB after a search, and 2.20 GB a
minute after, where it stayed (the idle trim lets the GPU's buffers go; the model stays, so the next
search does not load it).

## The sidebar squeezed by the columns (#31, macOS 15, 2026-10-25)

The reporter (macOS 15.8.1) saw the sidebar lose its padding while typing: 127pt wide, the section
titles against the window's edge. Reproduced in the 15.7 VM on 0.19.3 at their 1,020 x 682 window:
246pt at launch, 120pt the moment results showed. Tahoe never showed it, which is why it was not
reproduced there.
- CAUSE: before Tahoe, `columnHeader` stacks the header above the list, and the header's least width
  is the sum of its fixed columns (~900pt). That became the detail pane's minimum, and the split view
  took it from the sidebar, under the sidebar's own 230pt minimum. On Tahoe the header is an overlay
  and adds nothing to the pane's minimum.
- FIX: the header and the result rows are framed `minWidth: 0`, leading. Columns wider than the list
  run off the trailing edge, as in a too-narrow Finder list, on every system. On Tahoe this also fixes
  the same width: 0.19.3 centred the overflowing rows and header, so at 1,020pt the names and
  disclosure triangles sat under the floating sidebar ("rrisons Pri...", "Name" hidden). Where the
  columns fit (1,400 x 800) the two builds draw the same.
- A LAYOUT LOOP THE FIRST TRY FOUND: the header's trailing padding is derived from where the header
  ends (`headerMaxX`), measured inside its frame. Once the columns overflow, that end moves with the
  padding, so every pass added the padding again: the main thread never left layout (VM, typing
  stopped at the first word, `sample` all in `BrowserHeader.body`). At 2x the header's and rows' ends
  round to the same point and the loop stalls; at the VM's 1x they differ by a fraction and it does
  not. The header now measures the pane's edge, outside its frame, which the padding cannot move.
- Not reproduced: a narrow search field at launch - three relaunches in the VM with the window state
  kept (1,000 x 660 restored), the field full width each time.

## The search field squeezed by the centre item; Preview searched other words (#31, #32, 2026-10-25)

#31, the reporter's recording (macOS 15.8.1, 820 x 570, the narrowest the window goes): the field at
~170pt idle at launch, and in Recents ~33pt, its magnifier and clear button drawn over each other.
Reproduced in the 15.7 VM on 0.19.4 with the toolbar dump (which now prints the search field's own
frame: the search item has no `view`).
- CAUSE: before Tahoe the toolbar's free space is an empty `.principal` item, and AppKit keeps a
  centre item centred by shrinking the items on its right, the field first. The field got what the
  buttons beside it left of half the detail pane: 174pt idle in the empty window, 33pt in Recents
  (star, view picker, Search by File, Share). Focused, it takes its preferred width (240pt) and AppKit
  moves the centre item instead - which is why it was "sometimes" narrow: idle, not typing.
- FIX (macOS 14/15 only): the tuner holds the field at least at its preferred width (a 999 constraint
  on `searchField`). The centre item gives way idle as it did while typing. At 820pt the empty window
  shows 242pt with every button; Recents 242pt with the view picker, Search by File and Share in the
  overflow menu (all three are in the menu bar too). At 1,020 x 682 and 1,400 x 800 with results, the
  frames are identical to 0.19.4 (242 / 307pt): the hold only acts when the field would be squeezed.
- MEASURED AND NOT DONE: a 160pt floor keeps the view picker visible in Recents at 820pt but the field
  still jumps 162 -> 242pt on focus; no centre item packs the buttons left and leaves the field at 33pt
  in Recents.
- WHAT THE HOLD SHOOK LOOSE: OCR mode's empty page rail stayed open (3 of 3 runs; 0.19.4 1 of 4).
  Logged: the tuner set the sidebar to the rail's 168pt after SwiftUI had folded it, and setting a
  folded sidebar's divider unfolds it; the hold only made that the usual order. The tuner is now asked
  for no width while the drawer is folded (folded 3 of 3 after; back to search restores 260pt). A blanket "never touch a collapsed
  sidebar" was tried first and dropped: the launch correction is also what unfolds a sidebar the
  window restored folded.
- TAHOE runs none of the hold (no centre item); its window was compared with 0.19.4 side by side:
  820 x 570 empty and Recents, 1,400 x 800 with results, idle and focused, OCR mode folded and back -
  pixel-identical but for a row's indexed time. With a document open in OCR mode the rail comes back at
  the same width; 0.19.4's toolbar there settled differently from run to run (field expanded with
  items in the overflow menu, or collapsed with all of them shown), the new build the latter 3 of 3.

#32, the reporter on 0.19.4: double-clicking a passage opened Preview at the right page and highlighted
other words - "Adulterants in unregulated herbal" from the top of the page for a passage that starts
at the heading "THROMBOTIC MICROANGIOPATHY" lower down. The phrase was built from the passage's first
four words, looked for as a run of plain words on one line, and a comma among them, a heading on its
own line or a window starting mid-word missed, so it came from the top of the page. PagePhrase
(OmniKit now, with tests) finds the passage by any six of its words in order, punctuation and line
breaks aside, on its page or the next (a window of a long chunk), and takes the phrase from the
passage's own words first, at any length that qualifies (four words too when every earlier page is
checked), once on its page. `omni-verify pagephrase 220 <pdf>...` over 40 local arXiv PDFs, every
chunk opening and 220-character window (18,571 passages): the passage's own words for 81.9%, where
the 0.19.4 rule found the opening for 22.6% (the rest went to the top of the page). What is left is
mostly text no phrase can come from: reference lists, author lines, tables, prompts that repeat on
earlier pages. Preview, sent two of them as the app sends them: page 7 and page 21 of a 22-page paper,
the phrase highlighted at the passage, "Found on 1 page".

## Search history without a cap (Han, 2026-10-25)

Han: the sidebar's History keeps everything, and Settings > History > Keep history for has Forever.
The 200-row ceiling had filled with agents' searches: on his Mac all 200 rows were MCP queries from
the last 20 days, and his own typed searches were gone (one bookmark left).

MEASURED FIRST (`historybench:N` perf-script step, the app's own code on N synthetic searches over a
year, four in five from agents). Uncapped, the old code was linear with a heavy constant: every
element write re-ran the bookmark mark over the whole list (8 writes per recorded search), and the
merge, the suggestions and the day folders lowercased, parsed or sorted every row each time.

| per action, main thread | 10k old | 10k new | 100k old | 100k new |
|---|---|---|---|---|
| a typed search recorded | 105 ms | 2.8 ms | 1,066 ms | 30 ms |
| an agent's search recorded | 5.3 ms | 0.07 ms | 52 ms | 1.6 ms |
| suggestions, a keystroke matching nothing | 25 ms | 0.4 ms | 239 ms | 4.4 ms |
| the day folders | 20 ms | 1.0 ms | 202 ms | 0.7 ms |
| the bookmark mark, a keystroke | 2.8 ms | 0 | 28 ms | 0 |
| save (encode) | 30 ms | off main | 289 ms | off main |
| launch read | 139 ms | off main | 1,409 ms | off main (440 ms) |

HOW: the list is newest first by `lastUsed`, always (`placeHistory`, `touchHistory`), so nothing sorts
it and a day folder is a range found by bisection (only an open day's rows are copied). Each row
carries its text lowercased in memory (`folded`, not saved): suggestions are a `memmem` per row
with an early stop, the bookmark mark is a set kept per change that can touch a bookmark, and an
agent's repeated query is found in a set of the served rows' texts. The typed-search merge
compares typed rows only and parses just the rows with the same words.

STORAGE moved from the UserDefaults blob to `Application Support/Omni/history.json` (28 MB at 100k):
saved on its own queue, one write per burst (`HistorySaver`), flushed at quit; read off the main
actor at launch, the query-image sweep and the dead-recents prune waiting for it. The blob is read
once, collapsed (`canonicalized`, no longer run every launch), written to the file and removed. An
older Omni after this one sees an empty History. `-omni.historyFile <path>` gives an isolated run
its own file to read and write; with it: the real 201-row blob moved over, a typed, a REST and an
MCP search sent one second before Quit all in the file, the order newest first, a repeated query in
other case merged, nothing duplicated, a relaunch read it back, and 3-day retention left the 30
recent rows of the last three days and the bookmark.

## Renders per step: what re-ran the window (2026-10-26)

Han's goal: find the UI's lags, redraws and renders nobody asked for. THE INSTRUMENT: every `script`
line of the perf script now ends with `counts[...]`, the `UIProbe.count` tallies since the step
before (`UIProbe.takeTotals`, with `-omni.hangwatch`), and a temporary `UIProbe.count("body.X")` was
put at the top of all 125 view, modifier and commands bodies (a script inserts and removes them,
marked). New steps: `seedhistory:N` (the sidebar at N searches, left in place) and `note:<text>` (a
marker a harness waits on; the served-load loop runs between `note:served-on` and `note:served-off`).
The tour: an isolated instance over the eval tree plus 6,000 generated text files, indexing through
most of it, hangwatch at 100 ms; idle, typing, selection and arrows, grid and list, Find Similar,
the sidebar, browsing, Recents, served load at 4 searches a second, Settings, 20,000 searches seeded,
a history row clicked, served load again.

SIX CAUSES, each read off the counts and fixed:
- THE MENU BAR WAS PART OF THE APP'S BODY. Declared inline in `OmniApp.body`, every value a menu item
  reads (`menuSelection` on each arrow press, the bookmark state, Back and Forward) re-ran the App's
  body, and with it the window's root view, the toolbar and the search field: one whole-window render
  per selection change. `OmniCommands`, its own `Commands` type: select and arrows now re-run the
  menus only (ContentView 1 -> 0 a step).
- THE EMPTY STATE'S COUNT. "Search 6,226 files" read `indexedFiles` in ContentView's body, so while
  indexing every stats tick (1.5 s) re-ran the window. `IdleSearchPrompt` reads it: 5 -> 0 window
  renders in an 8 s idle step.
- THE SIDEBAR SELECTION IN THE ENVIRONMENT. Every row's `SidebarRow` modifier and the whole history
  section read it, so any click re-ran all of them: 462 rows and a 150 ms stall with 20,000 searches
  in History. One observed flag per row (`SidebarSelectionFlags`): a click re-runs the two rows
  whose flag flips (462 -> 1), the history row's tint comes from its own flag (`HistoryRowLabel`).
- A NEW OBJECT AS A @State's INITIAL VALUE MAKES EVERY RE-CREATION A CHANGED VIEW. The first form of
  the flags (`@State var flags = SidebarSelectionFlags()`) re-ran the sidebar with every window
  render, 18 times while typing one sentence: the initial value is built on each init and compared.
  The flags are made on first use. (`TypingTimers`, `SplitScroll`, `ResultFrames` and
  `ScrollZoomCatcher` have the same shape; their parents re-create them rarely, left.)
- THE PROGRESS PIE NEVER STOPPED ANIMATING. Progress is published up to 12 times a second and each
  value restarted the wedge's 0.2 s animation; Time Profiler on the idle window while indexing put
  most of the main thread in drawing those frames. `CloudSyncPie` takes whole percent (under a pixel
  of a 16 pt rim): pie renders 37 -> 10 an idle step; the idle main thread, old/new/old/new back to
  back, 2.41 / 1.74 / 2.41 / 1.97 s per 45 s.
- THE WINDOW READ THE RESULTS TO SEE IF THERE WERE ANY. `!model.results.isEmpty` re-ran it with each
  result set (the list, then the passages, then the grouping: 2-3 a search while typing), and the
  empty states read `isResolving`, which reads the query text: a window render per keystroke before
  the first result. `hasListedResults` (stored, written on change) and `EmptyResults` (its own view):
  typing "kitchen renovation" from an empty box, 11 -> 3 window renders.

- THE OCR WORKSPACE ASKED FOR THE WHOLE SECTION LIST TO SEE IF IT WAS EMPTY. `sectionIDs` reads every
  page's state, so `OCRView` re-ran with each page's every transition: 85 + 14 times over a 40-page
  scan's two 30 s windows. `hasSections` stops at the first section: 0 + 3. Same transcript (23,834
  tokens), 45.5 / 46.3 s; the main thread there is the streamed text and the readout (24 Hz, as
  designed), so its CPU did not move.

- THE HISTORY UNDER AGENT LOAD, KEPT FOREVER. With 20,000 searches over a year (Settings > History >
  Forever, `set:historyDays=...` then `seedhistory:N[:M]`, one in M bookmarked) and an agent searching
  4 times a second, the main thread blocked 27-30 times in 20 s, ~100 ms each (3.1 s, 15%): every
  batch of served searches re-ran the history section and SwiftUI's List diffed all its rows
  (`OutlineListCoordinator.diffRows` -> `ViewListTree.child(at:)`, which walks the rows from the
  start for each row: the cost grows with the square of the rows - 365 days a year, the bookmarks,
  the open days' searches). What fixed it:
  - served searches reach the history every 2 s instead of every 0.5 s, a quarter of the diffs (and
    the 400-search cap on a batch, written for the old 200-row history, is gone: it would drop
    searches at very high rates);
  - the row ids are STORED INTEGERS: the diff asks for every row's id many times over, and an id
    worked out on demand (a day's date formatted) was most of its time, a string id hashed
    Unicode-normalised at every visit (agents' queries carry CJK) much of the rest;
  - the days and the open days' searches are one flat ForEach, one view per element.
  At one bookmark in 50 (400): stalls over 50 ms under agent load 28 -> 0 and the main thread 3.75 ->
  2.38 s of 20 s; but a history click or a clear stalled ~340 ms flat against ~190 nested. At one in
  2,000 (10 bookmarks; Han has 1), flat and nested rows are level with the rest in place - 0 stalls
  under agent load in both, 1.7-2.1 s of main thread, the first fill ~330-350 ms - and clicks and
  clears are slightly better flat (3.0 / 3.1 s of stalls over six actions, nested 3.3 / 3.4): kept.
  Still left: the diff itself; fewer rows is the only further lever (older days folded into months,
  as Notes and Safari do), a design decision not taken.

WHAT IS LEFT WHILE TYPING, profiled (Time Profiler over five typed queries, ~150 keystrokes, a settled
index, hangwatch at 60 ms): blocks of 60-100 ms and one of 150 ms, each where a result set lands; the
time is SwiftUI's graph update and the window's layout (NSHostingView.layout runs inside an
NSAnimationContext group of SwiftUI's own, a third of the busy time), the toolbar's layout ~16%, no
Omni frame in the heavy stacks. Grid and list switch under 100 ms in the final tours. These are the
first-results costs testing.md ("UI stress") already took apart and staged over frames.

THE WHOLE TOUR, main-thread CPU, the build before and after back to back on the same script: 12.1 s
-> 9.5 s on the GPU. In Neural Engine mode (the programs from a warm cache used in place: a COPIED
cache recompiles, the compiled forms follow the path) the same counts fall the same way, and the
steps that idle or serve while indexing take 25-40% less main thread (an idle 8 s 389 -> 231 ms,
12 s of served load 1,323 -> 985 ms). The menus follow the state through `OmniCommands`
(`menustate:<menu>` step: Open, Quick Look and the rest turn on with a selection, Ignore Folder
names the selected file's folder, Back turns on after a search, all off after a clear), and five
states (idle, a result selected, a history row selected, a folder selected, cleared) are
pixel-identical to the build before. Unchanged on purpose: the prompt's rolling digits (0.3 s per stats tick, the design), the
history section re-running once per batch of served searches (about 1.6 a second under agent load,
~15 ms each at 20,000 searches), the first results' toolbar insert (testing.md, "UI stress").
