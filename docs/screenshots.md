# Screenshots (2026-11-10)

Omni takes a shot of the desktop while the Mac is in use and keeps every shot as a frame of a
video, and as nothing else. The few worth finding again are in the index under a path that spells
their moment; no file is behind that path, and the frame is taken out of the video when a search,
a thumbnail or an agent asks for it (0.20.3, "Video only"; through 0.20.2 each indexed shot was
also a picture file). Search, filters, thumbnails, Quick Look, Find Similar, MCP, CLI and REST are
the existing pipeline: a shot rides it through a content source as a Photos asset does
(`ScreenContentSource`), a picture embedding plus `ScreenText`. No schema change, no new model, no
new package.

The one thing served surfaces add is time: a screenshot result says when it was taken and what the
stretch of work around it was, and `search` lists the files changed in a time range with no query.
That is the chain an agent follows: query -> a screenshot -> its time -> the files changed then.

## Decisions

- Opt-in, like the clipboard. Off until turned on from the sidebar row or Settings > Files; macOS
  then asks for Screen & System Audio Recording, which only the user can grant.
- A shot follows input, not a clock (`ScreenRecorder`): one every 5 s while there is input (keys,
  pointer, scroll: `CGEventSource.secondsSinceLastEventType`, no permission needed), one more when
  input stops (marked settled), then a look at 30 s that doubles while the screen stays as it was
  (30, 90, 210 s). A screen that moves by itself (a build, a video) keeps the 30 s.
- Nothing is taken from 5 minutes after the last input until the next input, whatever the screen
  does (`ScreenCadence.away`; "Nobody at the Mac"). Something watched hands-off, a film or a
  meeting, is recorded for those 5 minutes only.
- No input and a still screen writes nothing: a shot whose signature equals the last stored one is
  not stored (`ScreenHistory.record`).
- Every stored shot is in the timeline; few are in the index (`ScreenGate`). A shot is indexed
  when at least 10% of its tiles both changed and show something, against the nearest of the last
  8 indexed shots; a settled shot at 0.5%. A flat screen is never indexed.
- Not taken: screen locked, another user at the console, screen saver, display asleep. Omni's own
  windows are left out of the shot (a search in Omni would otherwise be found by the picture of its
  own results). The pointer is not drawn. Windows an app marks as not shareable are left out by the
  system. Secure input does NOT hold a shot back (it did through 0.20.2: "The recorder that stopped
  and said nothing").
- A look that fails is told: "Not recording" with the reason in Settings and a badge on the sidebar
  row, from the second failed look in a row, gone at the first that works (`ScreenLookHealth`).
- Retention 30 days by default; 7 / 30 / 90 days / forever in Settings, with the space used
  (the video). Clear is on the sidebar row's context menu.
- Video only (Han, 2026-11-12): "the datetime serves as a pointer. only at search time ... retrieve
  the pointer, use the pointer to retrieve the frame and return". No folder of pictures to browse in
  the Finder; a shot becomes a file only when something that takes nothing else asks (a drag, a
  copy, Share, Quick Look, a linked Mac's fetch). "Reveal" opens the timeline at that moment. The
  update deletes 0.20.2's picture files. The video stays as it was: HEVC quality 0.7, a key frame
  every 30 shots.
- Apps left out (Settings > Files > Screenshots, "Excluded apps", the group's last row): their windows are not in a
  shot, and no shot is taken while one of them is in front.
- Served to this Mac's agents (MCP, CLI, REST, Shortcuts) as the user's files are, and to Linked
  Macs as the rest of the index is (Han, 2026-11-10; the first build kept them back with the
  clipboard, which alone stays in `AppModel.unsharedFolders`).

## Parts

1. `Sources/OmniKit/ScreenChange.swift`: `ScreenSignature` (green channel on a grid about 960
   wide, tiles of 16 x 16 samples, which tiles show something) and `ScreenGate`.
2. `Sources/OmniKit/ScreenCodec.swift`: VideoToolbox HEVC on the hardware encoder, quality 0.7, a
   key frame every 30 shots, no frame reordering; the decoder; the HEIC writer a shot leaves the
   app through.
3. `Sources/OmniKit/ScreenHistory.swift`: one folder beside the index.
   `Application Support/Omni/Screen History` holds the segments (`<start>-<display>.omniscreen`,
   one per display per hour or 720 shots, a new one at every launch): records appended one at a
   time, each with its time, the app in front, its window's title and the HEVC frame. Never
   crawled. A record cut short by a kill is the last thing in its file and a reader stops at it:
   nothing to repair. An indexed shot is a flag in its record and a path in the index,
   `<Application Support/Omni/Screenshots>/2026-11-10/2026-11-10 13.16.27 Editor - source.swift.heic`:
   the path 0.20.2's picture file had, spelled from the segment's offset from UTC (its header)
   and the record (time, app, title, and which of the names of one second is its own), with no
   file and no folder behind it. The index reads the frame at a pixel a point of the display
   (`ScreenHistory.stillSide`, kept in the header; 0.20.0 read at most 1568 wide, which a desktop
   wider than that could not be read from: "The reader"). `shots.catalog` beside the segments
   lists the indexed shots, so a pass does not walk the video ("Video only").
   `Sources/OmniKit/ScreenShots.swift`: the registry the indexer finds a history by, and
   `ScreenContentSource`, which answers for a shot what a file system answers for a file.
4. `App/ScreenRecorder.swift`: one `SCScreenshotManager` still per look and display (not a stream:
   a stream delivers frames whether or not anything moved), on its own queue.
5. `App/ScreenTimeline.swift`: the Screenshots place. One day on a track true to time, a point a
   shot (indexed ones larger), a playhead to drag, arrow keys, a day picker, the shot's time, app
   and window. A decoded shot goes straight into a layer, so a drag draws frames with no SwiftUI
   render. Off and empty it is the standard empty state with Turn On. Without the grant (recording
   on, macOS saying no) the place, the sidebar row's badge and the Settings row say so and open
   Screen & System Audio Recording; the shots already held stay on the track, where the badge is
   what says it.
   The track zooms down to two minutes across: a pinch or a vertical scroll about the pointer,
   Command-= / Command-- / Command-0 and the track's menu (Zoom In, Zoom Out, Show Whole Day)
   about the playhead; a sideways scroll pans. A shot drags out as a PNG
   of the whole frame, a real file written when the button goes down (a promised file is taken by
   the Finder and Mail, not by a browser's upload area or a chat window). Delete (the shot's
   menu; Command-Delete for the one shown) removes this screenshot, this session or this day after a
   confirmation that names the count: the frames, then their rows (`ScreenHistory.delete`:
   the frames after a deleted one, up to the next key frame, are encoded again from a new key
   frame, the segment written beside itself and renamed over).
6. `App/Serving/ScreenHints.swift`: what a served screenshot result says (below).
7. AppModel: the history's root joins `crawlRoots` while there is a history, outside
   `addedFolders`; a pass takes its files from the history and not from a crawl
   (`Indexer`, `shotSources`). `App/ShotActions.swift`: a shot as a file, for what takes nothing
   else.

## Served: a time, a stretch of work, and the files changed then

- A screenshot result carries its time to the second with the local offset
  (`2026-11-10T13:49:13-08:00`); other results keep the short date.
- Under the results, one line per stretch of work the shown screenshots fall in (shots with no
  gap over 5 min: `ScreenHistory.session`), not one per result:
  `Screenshots 1, 2, 3: taken while the Mac was in use from ... to ..., in Editor (source.swift),
  Browser (Field notes). Files changed then: search with modified_after "...", modified_before
  "..." and no query.` The CLI gets the same line with `omni search --modified-after ...`.
- Structured rows (REST, `omni search --json`) carry it as a `screenshot` object: `taken`, `app`,
  `title`, `session {start, end, apps [{name, windows}], modified_after, modified_before}`.
- `search` no longer requires `query`. With `modified_after` or `modified_before` and no query it
  lists the files last modified in that range, newest first, through the store's existing
  filter-only listing (`VectorStore.listMatching`): no embedding, no score, no grouping, this Mac
  only, not recorded in the search history. The clipboard is left out as everywhere served; the
  screenshots are left out unless `folder` names their folder, since the question is what was
  changed, and the shots themselves are what led there. With neither a query nor a range it is an
  error that says so. `omni search` shows `[<query>]` as optional.
- Linked Macs. A linked Mac's screenshots are results like its other files, and the Mac that
  answers attaches what only it knows: the moment and the stretch of work of each
  (`LinkWire.SearchReply.shots`, plain values; the asking Mac holds no history of another's screen).
  The session line then names the Mac, and its call carries it: `Screenshot 1 on Studio: taken
  while Studio was in use from ... Files changed then: search with modified_after "...",
  modified_before "...", macs ["Studio"] and no query.` A range with no query goes to a linked Mac
  only when `macs` names it: the files changed in an hour on this Mac are not the question "what
  changed everywhere", and a listing has no score to cut a second Mac's files by. The answering Mac
  lists through the same `ServingBackend.list` (`LinkWire.SearchRequest.isListing`, reply
  `listed`); a Mac whose Omni is older says so ("its Omni is older and cannot list files by time").
  Not linked: the timeline (a Mac's history is browsed on that Mac), and a linked Mac's Screenshots
  folder as a place in the sidebar.
- A listing is by LAST modification time: a file changed in the range and again later is listed
  at its later time and falls out of the range. The index keeps one date per file.

## Measured (M3 Ultra, Xcode 27.2, 2026-11-10)

`omni-verify shotbench <source> <prose> <photo> [--scale 1|2]` draws a 150-shot session (typing,
scrolling, browsing, a blinking caret, an app switch) and puts it through the app's own encoder,
gate and reader. 1920 x 1080 unless said.

Codec, per shot:

| | KB | encode ms | decode ms | cold jump ms | PSNR dB |
|---|---|---|---|---|---|
| HEVC q0.7, key 30 (shipped) | 20.5 | 7.6 | 2.3 | 74 | 38.2 |
| HEVC q0.5, key 30 | 10.5 | 7.5 | 2.7 | 59 | 34.6 |
| HEVC q0.8, key 30 | 26.4 | 7.3 | 2.3 | 59 | 39.1 |
| HEVC q0.7, key 10 | 24.5 | 7.7 | 2.2 | 24 | 38.5 |
| HEVC q0.7, key 60 | 20.2 | 7.6 | 2.1 | 134 | 38.1 |
| HEVC q0.7, no keys | 19.0 | 7.8 | 2.3 | 343 | 38.1 |
| HEIC q0.8, each alone | 101.7 | 35.4 | 28.0 | 28 | 39.1 |
| HEIC q0.6, each alone | 78.5 | 35.4 | 29.2 | 29 | 37.8 |
| PNG, each alone | 251.8 | 13.4 | | | |

- As video the session is 3.15 MB against 15.63 MB as HEIC q0.8 at the same PSNR band and 37.8 MB
  as PNG: 5.0x and 12x. On the typing stretch a shot is 5 KB against 103 KB (about 20x): what
  changed is a line. Encoding is 4.7x faster than a HEIC.
- Key 30 over key 60 or none: the cold jump (decode from the key frame to a shot) is what a click
  on the track waits for, 74 ms against 134 and 343, for 1.5% more bytes than key 60.
- 3840 x 2160: HEVC q0.7 key 30 is 42.2 KB a shot, encode 25.2 ms, decode 5.5 ms, cold jump
  161 ms, 41.8 dB; HEIC q0.8 235.4 KB; PNG 638.6 KB.
- Signature 0.38 ms a frame, compare 0.33 ms a pair. The reader: "The reader" below.

Gate (indexed of 150, share of the session's distinct words still found in what was indexed):

| threshold | indexed | words held |
|---|---|---|
| 2% + settled | 55 | 96.1% |
| 5% + settled | 49 | 95.7% |
| 10% + settled (shipped) | 49 | 95.7% |
| 10%, no settled shot | 47 | 95.6% |
| 20% + settled | 34 | 92.4% |
| 35% + settled | 4 | 38.5% |

At 3840 x 2160 the shipped gate indexes 49 and holds 95.0%. The caret stretch indexes none at any
threshold. The curve is flat from 5% to 10% and falls after; 10% is the last point before it does.

In the app (isolated run fed the same session, Nano): 150 stored, 47 indexed; the segment
3,047,740 bytes (5 key frames of mean 91,791, deltas of mean 17,801); the 47 files for the index
5.30 MB, about 113 KB each at a pixel a point (0.20.0: 3.9 MB, 83 KB at 1568 wide), so the indexed
pictures take more disk than the whole video. A scrub of 349 steps: 347 decodes, p50 2.6 ms, p90
22.2 ms, max 54.0 ms.

Renders per step on the timeline (`-omni.hangwatch YES`, 0.20.1): opening the place draws
ScreenTimeline 3 times, the bar once and each part of it twice; a selected shot draws the playhead,
the count and the label once each and neither the picture's pane nor the track (4-6 ms of main
thread); a zoom or a pan draws the playhead, the track and its points once (15-24 ms); a deleted
shot ScreenTimeline twice and the five parts once (14 ms); a scrub of 71 shots draws playhead,
count and label 71 times each in 138 ms of main thread in all.

Served:

- The hint (`shotbench hints <dir> 30`: 30 days, a shot every 5 s for 8 h unbroken, 172,800
  shots, 240 segments): the stretch around a shot 0.79 ms warm (p90 0.86), 3.1 ms the first time
  a day's segments are read (max 9.7). Thirty screenshot results walked one by one were 22.6 ms;
  a stretch is walked once however many results fall in it (`sessions(around:)`): 2.5 ms for
  thirty over three days.
- The range listing (`shotbench listrange <clone>/index.sqlite`, the 9.73M-row bench index).
  0.20.0 walked every resident row on the store queue, 160-194 ms whatever the range was, and a
  search that arrived during one waited it out. `VectorStore.listMatching` now walks by file (2.7M
  against 9.73M rows): a range of 13 min to a day about 10 ms, 30 days 27-28 ms, no filter 76 ms, a
  served filter with no range 95-97 ms; the same files in the same order as before (digests equal).

## The reader (0.20.1)

0.20.0 READ EVERY SCREENSHOT WHOLE AT 1568 ON ITS LONG SIDE, and wrote its stills at that size. A
desktop wider than 1568 points was read at less than a pixel a point, and its text was not there
to find: of an editor's words 97% on a 1512-point desktop, 80% at 2560, 29% at 3008, 1% at 3840.
The same held for every screenshot file in the index, not only Omni's own. Found while measuring
"displays above 4K"; a quality bug in the existing reader, fixed there (`ScreenText`).

Now: a screenshot is read from its file at 1.5 pixels a point (points: long side over dpi / 72;
enlarged with high-quality interpolation when the file has fewer pixels; 32 million pixels at
most), in 1024 x 1024 pieces that share at least 196 pixels. One that fits one piece is read whole.
A still is written at a pixel a point. Scans are read whole as before (a page at 1108 x 1568 with
14-22 pt text: 99-100%).

Chosen on 48 drawn pages at 1920 x 1080 (words read): whole at 1568 44.2%, none at all on 20; a
1.0 still enlarged to 1.5 in 1024 pieces 96.6%; pieces of 784, 1176 or 1568 and densities of 1.25,
1.75 or 2.0 all read less. On 38 real screenshots: 79.0% of confirmed words whole (20 files under
90%, the worst 45.3%), 94.9% in pieces (2 files, the worst 88.9%).

Through the app's own code, `omni-verify shotbench stillsize <source> <prose> <photo> [desktop]`
(4 editor frames a desktop; lines = drawn lines of 12+ characters read exactly):

| desktop | arm | read at | pieces | words | lines | ms | KB a still |
|---|---|---|---|---|---|---|---|
| 1512 x 982 @2x | 0.20.0, whole | 1568 | 1 | 97.3% | 75.7% | 172 | 137 |
| | still, in pieces | 2268 | 6 | 98.4% | 79.9% | 723 | 129 |
| 1920 x 1080 @1x | 0.20.0, whole | 1568 | 1 | 96.7% | 76.6% | 132 | 100 |
| | still, in pieces | 2880 | 8 | 97.4% | 73.4% | 789 | 136 |
| 1920 x 1080 @2x | 0.20.0, whole | 1568 | 1 | 97.4% | 83.5% | 134 | 105 |
| | still, in pieces | 2880 | 8 | 98.0% | 78.5% | 777 | 140 |
| 2560 x 1440 @1x | 0.20.0, whole | 1568 | 1 | 80.0% | 34.4% | 203 | 87 |
| | still, in pieces | 3840 | 15 | 97.2% | 79.4% | 1249 | 182 |
| 2560 x 1440 @2x | 0.20.0, whole | 1568 | 1 | 77.1% | 37.6% | 186 | 91 |
| | still, in pieces | 3840 | 15 | 98.0% | 81.2% | 1242 | 189 |
| 3008 x 1692 @2x | 0.20.0, whole | 1568 | 1 | 29.3% | 4.6% | 159 | 81 |
| | still, in pieces | 4512 | 18 | 97.6% | 76.6% | 1333 | 220 |
| 3840 x 2160 @1x | 0.20.0, whole | 1568 | 1 | 1.1% | 0.0% | 96 | 66 |
| | still, in pieces | 5760 | 28 | 96.9% | 74.8% | 1867 | 270 |

Reading the native frame instead of the still (the third arm) is within 0.7 points of words on
every desktop, so the still at a pixel a point is enough and no second file is kept. The lines
still missed at 1920 x 1080 are the reader's own misreads of punctuation ("{" as "1", "->" as ">"),
in the whole read as well. The eval bed's nine screenshots (1280 x 800; English, Chinese,
Japanese; 38 known lines): 37 whole, 37 in pieces, a different line each. Han's 38 real
screenshots: 7,257 words whole, 7,614 in pieces (`shotbench reader <paths>`, counts only).

PUTTING THE PIECES BACK (`ScreenText.recognize(pieces:)`). The first merge read 68-76% of lines at
96-98% of words: it broke lines. Three things, each found from a missed line:

- Vision boxes a Latin word, not its letters (`boundingBox(for:)` of one letter is the word's);
  Chinese and Japanese characters have boxes of their own. A Latin word is taken whole from the
  piece that has all of it; one longer than what the pieces share and cut by both edges (an
  identifier, a URL) is laid together from both readings (`stitch`). CJK is split at the seam.
- Vision gives one drawn line as several observations ("import", "SwiftUI"), never exactly level:
  sorted by height they came out scrambled. Runs are grouped into rows and ordered left to right.
- A box moves 1-3 pixels from one piece to the next: what is within 4 pixels of a seam is kept by
  both pieces and matched by place, every shared word, not only the last and the first.

WHAT IT COSTS. A still at a pixel a point is larger: the 150-shot session fed to the app writes 47
stills of 5.30 MB, 113 KB each (73-171), against 3.9 MB and 83 KB at 1568 wide; the video is the
same 3,047,740 bytes. Indexing those 47 end to end (`omni-verify imgindex <nano> <folder>
[--whole-screens]`, three runs each, interleaved): 10.9-11.2 s read whole, 41.5-42.0 s in pieces, 0.23 s
against 0.89 s a screenshot. All of it is Vision, which reads one picture at a time whoever asks:
2, 4 or 8 pieces at once took 41.1, 41.2 and 41.0 s against 42.0 (arm retired). The gate indexes
a shot about every 16 s of use, and the reader holds neither the store queue nor the engine gate.

The upgrade: a screenshot's content key and picture chunk key carry `|t2`, its ledger key `|r2`,
and the enrichment scope is s4, so every screenshot an index holds is read again once, by the
existing enrichment pass (`enrichOne`: the picture's vector kept under today's key, the text
beside it replaced, by nothing when there is none now). No schema change. `ScreenText.inPieces = false` is the 0.20.0 read, for tests.

Through the app, on an index a 0.20.0-reader build made (32 stills at 1568 wide, isolated): opened
with this build, the pass names 32 candidates and reads them in 22.2 s (0.69 s each); 49 text chunks
become 60, the 32 pictures keep their vectors, `storeaudit` 0 failing checks; opened again, no
candidates and nothing read. The same index killed (SIGKILL) 4 s, 7 s and 2 s into the pass, then
left to finish: the same text, keys, ledger and search digests as the run that was not killed
(`searchreal` 32c49651d5a8d6be), 0 failing checks.

The labeled eval, Nano, a fresh index each (`omni-verify eval ... --rebuild`, `--whole-screens` for
0.20.0's read): ALL MRR 0.896 both, screenshot 1.000 and shot-exact 1.000 both, every one of the
155 queries at the same rank. The bed's screenshots are 1280 x 800, which 0.20.0 read well.

## The still that cost the video its reference

ImageIO's HEIC writer on the hardware encoder starts an HEVC session per picture, and the video
session's next frame came out a key frame: every indexed shot was followed by one, and the session
grew to the size of its stills. `ScreenCodec.stillOptions` asks for the software encoder
(`kCGImageDestinationUseHardwareAcceleration: false`). At 1568 x 882, q0.8: hardware 48,613 bytes
and 40-45 ms (84 the first); software 49,802 bytes and 33 ms.
`testTheShotAfterAnIndexedOneIsStillSmall` holds it.

Why HEIC: the six real stills (1568 wide, q0.8) are 71-86 KB; the same pictures as JPEG q0.8 are
152-164 KB (1.9x) and as PNG 307-319 KB (3.8x), re-encoded from the stills with `sips`. The stills
are about half of what the feature stores. The cost: a HEIC is a file most agents cannot open
(of the model APIs only Gemini's takes one; Claude Code's reader refuses it). So the file stays a
HEIC and the picture is converted when asked for: `read_file` on an indexed image answers with a
JPEG whatever the file is, a screenshot up to the display's own size from the history
(`ScreenHistory.frame(ofStill:maxDimension:)`), and `omni` saves each picture a command returns and
prints its path (docs/CLI.md). MCP `search` inlines image hits as JPEG thumbnails (800 px).

## A real session (2026-11-10)

An isolated development build with `-omni.screenshots.capture YES` on Han's Mac: one display,
1920 x 1080, the grant of the terminal it was started from (no prompt). First with nobody at the
Mac (the idle look), then a working session.

- Capture 46-111 ms. Record 8-11 ms for a delta, 87-280 ms when a segment starts or a still is
  written. A key frame 72-88 KB, an idle look's delta 0.6-3.4 KB at novelty 0.005-0.009 (stored,
  not indexed), a still 71-86 KB.
- Footprint with the engine quiet for 70 s: 2017 MB with recording off, 2021 MB on. Timeline open,
  a 65-step scrub, closed: within 9 MB. An indexed shot is an image through the engine, which
  holds about 585 MB more for about 60 s after any image.
- Scrub on real shots: 63 decodes, p50 3.7 ms, p90 11.9, max 16.9.
- A search in the window with no scope returns a real shot first among ordinary files.

A WORKING SESSION, 464 s (2026-11-11). The app in use was a second Omni in front at 1500 x 950 (a
renamed copy: "Isolated runs"), driven through its script: a new search every 16 s, selections,
arrow keys, list and grid, folders. Pointer events that move nothing kept the session's idle time
under 5 s. The recorder's log against the script's marks:

- In use, 206 s: 41 looks 4-6 s apart, 36 stored (5 the frame before, not stored), 6 indexed
  (novelty 0.113-0.195; the others 0.013-0.099). Video 562,973 bytes, 13.7 KB a look; 6 stills
  544,045 bytes.
- Input stops: the last shot 2 s in, the settled one at 7 s (indexed at 0.098), looks at 37 s and
  72 s (0.001, stored, not indexed).
- Input on a screen that does not change, 61 s: 12 looks, 6 stored at about 0.6 KB each, 6 not
  stored, none indexed.
- An excluded app in front, 35 s of use: no shot. Allowed again: a shot within 1 s (indexed, 0.107),
  then 7 looks in 35 s, 6 stored, 3 indexed.
- An excluded app behind another one: `screen-targets displays=1 left_out=1`, its window gone from
  the shot (0.243, indexed), its turning dark not seen (0.006 three times), back when allowed
  (0.274, indexed).
- The whole session: 68 looks, 56 stored, 12 indexed, 3 of them settled. Video 867,207 bytes, 12
  stills 1,068,806 (89.1 KB each). Capture p50 58.5 ms (max 81), record p50 10 ms (max 52, a still
  written). Footprint 2.1-3.1 GB, median 2.6: an indexed shot is an image through the engine.

A GRANT TAKEN AWAY WHILE OMNI RUNS was only logged: the capture failed, the targets were looked up
again and the next tick tried again, every 5 s, with nothing in the window saying so. Now a capture
or a content fetch that macOS answers with `SCStreamError.userDeclined` is no access
(`screen-access allowed=false`): the pane when no shots are held, the Settings row, and on the
sidebar row the badge a folder macOS denied has, which opens Screen & System Audio Recording. The
recorder asks again every 5 minutes instead of every tick, and the first frame that arrives ends
it. macOS's own answer takes a change to the user's privacy settings to provoke, which a test
never makes: perf-script `shotdecline:<seconds>` enters the state at a tick, as the answer does.
Through the app on the real screen: 173 shots held, declined, no access at the next tick, no look
with the Mac in use until the retry 30 s later, the held shots on the track, the badge on the row;
then the retry's frame, access at the next tick, a look every 5 s again, the badge gone. The row
renders once when access goes and once when it comes back, and not at all between
(`body.ScreenshotsRow`, per script step); no main-thread block over 250 ms.

THE SYSTEM'S RECORDING INDICATOR AND ITS RECURRING PROMPT are macOS's own; Omni neither draws nor
hides them, and a no to the prompt is the grant taken away, above.

SEVERAL DISPLAYS, fed (this Mac has one): three in one session across midnight, one of them
3840 x 2160, 116 frames. A segment per display, every frame stored, 39 indexed. The earlier day
lists three displays (47, 27 and 6 shots), the later one two (33 and 3); the day's menu switches
between them. A query for words shown only on the 3840 display returns its three shots first
(stills 3840 x 2160, 194-216 KB) and the session line runs 23:56:05 to 00:02:40 across the day. No
main-thread block over 250 ms.

EVERY START INDEXED THE SAME SCREEN AGAIN. A recorder that starts (a launch, Turn On) had an empty
gate, and an empty gate admits anything: four of the first five real stills were one screen. A
display's gate now starts from the shots of it the index holds (`ScreenHistory.gate(for:)`: the
last 8 indexed shots whose still is on disk, decoded from the video, at most 24 segments back).
Through the app on the real screen: off then on, the first shot stored and not indexed at novelty
0.005; a relaunch, 0.006; a screen that had changed while it was off, 0.176 and indexed. A still
that came back through the codec adds no novelty that shows beside the screen's own 0.005-0.009.
`testARecorderThatStartsAgainDoesNotIndexWhatTheIndexHolds`, `testAScreenWhoseFileIsGoneIsIndexedAgain`.

A SHOT AFTER TURN OFF. A capture in flight when recording was turned off recorded into the stopped
history 50 ms later, with an empty gate: indexed. The completion now drops it when the timer is
gone (`screen-shot dropped`). Provoked: on, 0.9-1.5 s, off, nine times; eight landed in a capture
and each was dropped, no frame and no still written.

THE HARDWARE ENCODER GIVES UP ITS REFERENCE WHEN ANOTHER ONE STARTS OR ENDS, in this process or any
other, HEVC or H.264: the next shot is a key frame, correct and the size of one. `omni-verify
shotbench reference <frames dir>` (frames 0-5, the arm, frames 6-7; K key, d delta, bytes):

| between frames 5 and 6 | frames 6-7 |
|---|---|
| nothing, or 30 s of nothing | d1568 d1072 |
| decode what was written (the timeline's scrub) | d1568 d1072 |
| a HEIC still, software (shipped) | d1568 d1072 |
| a HEIC still, hardware | K77760 d1384 |
| a second hardware encoder, same process | K77760 K78529 |
| an HEVC or H.264 encoder in another process | K77760 d1384 |

An encoder that was already running and goes on encoding costs nothing, and Omni's shots cost
another app's encoder nothing (`shotbench neighbour`: its 12 HEVC or H.264 frames are the same
bytes with a shot of ours between every two). So a call or a recording that starts costs one key
frame, and a test run that makes encoders beside a recording instance spoils its byte counts.

The software encoder (RETIRED, numbers kept): immune in every arm above. 150 shots at 1920 x 1080,
q0.7: 3.25 MB against 3.57, 36.4 ms a shot against 7.7 (slowest 168 against 92), PSNR 37.0 against
37.2, text read back 86.9% against 80.0%. At 3840 x 2160: 7.01 MB against 7.30, 81.2 ms against
24.4 (slowest 360 against 91), 40.6 dB against 41.0, 90.1% against 88.1%. Its work is in
VTEncoderXPCService: 3.15 s of CPU for 40 frames at 1920 x 1080 against 0.07 s, 79 ms a shot
against 2, every shot of every day, to save a key frame when another encoder starts. Hardware stays.

AN ISOLATED RUN'S LINKED FILES WENT TO THE USER'S CACHE (same bundle id). They go beside its index
now (`RemoteFiles.root`).

Thirty days, from the working session: unbroken use is 716 looks and 105 indexed shots an hour,
9.84 MB of video and 9.5 MB of stills, 19.3 MB an hour: 4.6 GB for 240 hours at 1920 x 1080. The
session's own mix, pauses included, is 15.0 MB an hour: 3.6 GB. Time away from the Mac costs
nothing ("Nobody at the Mac"). Settings shows the space really used.

## Nobody at the Mac (2026-11-11)

Han's laptop, at home with an external display that never sleeps, nobody at it for most of the day:
2,321 shots and 348 MB in 12 hours, about one stored shot a minute through the hours away.

The idle clock was right. The policy recorded: with no input 0.20.1 looked every 30 s, stored the
look when any tile differed from the last stored shot, and a stored look kept the 30 s. A menu bar
clock changes the screen every minute, a page that refreshes more often.

IS THERE AN API THAT KNOWS A PERSON IS THERE? No. macOS has no presence sensor an app can read;
every idle signal is the time since the last input event, read through different doors:
`CGEventSource.secondsSinceLastEventType` (`.combinedSessionState`, `.hidSystemState`), IOKit's
`HIDIdleTime`, powerd's `UserIsActive` assertion, and powerd's user-activity level (private:
`IOPMGetUserActivityLevel`, what `pmset -g useractivity` prints). ActivityWatch (HID idle and the
lock, away after 3 min) and Chromium (`idle_mac.mm`: the combined session state, the screen saver
and the lock) use the same clock. Measured on this Mac:

- One synthetic pointer event posted to the session reset all of them together, `.hidSystemState`
  and `HIDIdleTime` included: none is hardware-only, and an app that posts events is a person to
  every one of them.
- powerd's level went from "present, active" (0x01) to "present, passive" (0x06) at idle = 301 s:
  its timeout is 5 minutes (`kIOPMDefaultUserActivityTimeout`). After 7.6 h without input, with the
  display held awake, the level still read "present, passive": passive is any app holding the
  display on, not a person.
- `IOPMAssertionDeclareUserActivity` from another process put the level back to "active" while the
  input clocks went on counting: the level is also whatever an app declares.

So the input clock stays (`.combinedSessionState`), and the rule changed: `ScreenCadence.away`,
300 s after the last input, nothing is taken until there is input again, whatever the screen does.
A capture macOS declined is not retried while away either. The cost, taken knowingly: a film or a
meeting watched without touching the Mac is recorded for 5 minutes after the last touch.

The cadence is a value type in OmniKit now (`ScreenCadence`, `ScreenCadenceTests`, 7): a screen
that changes by itself is looked at 9 times in the 5 minutes and never after, over 12 hours; the
negative control (idle held under `away`) is 119 looks an hour; the user back is shot at the
first tick. (In the app the 30 s counts from the look coming back and the tick is 5 s: a look
every 35 s, 8 in the 5 minutes.)

Both builds beside each other on the real screen, nobody at the Mac, a screen that changes by
itself, one input event at 0:00 and none after:

| | 0.20.1 | with `away` |
|---|---|---|
| stored, first 5 min | 10 | 10 |
| stored, the next 8 min | 15 | 0 |
| started with 408 s of no input | a shot at once | nothing (`screen-away 1 idle=408`) |

A look's delta was 0.9-2.6 KB at 1920 x 1080, so the hours away were most of the laptop's shots
and little of its bytes: the 348 MB is the hours in use, on two displays at their full pixel
count (a 3840 x 2160 shot measured 2.0x the video and 2.6x the still of a 1920 x 1080 one, and
1920 x 1080 in unbroken use is 19.3 MB an hour). The laptop's own split is not measured: it was
the size of its `Screen History` (video) and `Screenshots` (stills) folders, and from 0.20.3 there
is only the first.

## The recorder that stopped and said nothing (2026-11-12)

Han's laptop, 0.20.1: the last shot is 22:01:33 on 2026-11-11, of a password prompt. The next
morning, on 0.20.2, the timeline was still on that day, and turning recording off and on stored
nothing. Settings showed no "no access" row. Turning recording on forgets the last stored frame
and the gate, so the first look after it is stored whatever the screen shows: something stopped
every look before it was taken, or every look failed, and neither had a place in the UI. The
laptop was not in reach; the cause below is inferred from what the build could do, not read off it.

SECURE INPUT IS ONE FLAG FOR THE WHOLE LOGIN SESSION. `IsSecureEventInputEnabled()` is true in
every process while any process holds it, and a process that never lets go (a prompt left open, an
app that crashed holding it) holds it until it exits. Through 0.20.2 the recorder took no shot
while it was true. A helper with no window held it for 2 s and then for 22 s on this Mac: every
process read true, and the session's `kCGSSessionSecureInputPID` named the app in front both times
(35778, then the isolated Omni), never the helper: who holds it cannot be asked, so "only while the
holder is in front" cannot be built. The stop is REMOVED: a password field draws dots, in the shot
as on the screen, and an app to keep out is an excluded app. With the 22 s hold and input every
2 s, this build stored 4 looks.

EVERY OTHER WAY A LOOK FAILS IS TOLD. Before, a capture error, a missing frame, an encoder that
would not start and a write that failed were perf-log lines; an unanswered `SCShareableContent`
or capture call left `inFlight` set and every later tick returned at its first line, for good.
Now:

- A look ends as worked or failed with a reason (macOS's own error text, or "macOS listed no
  displays.", "macOS returned no frame.", "The frame could not be read.", "The video could not be
  started.", "The frame could not be encoded."). `ScreenHistory.lastFailure` tells a screen that
  did not change from a shot that could not be stored. A look that fails on one display and works
  on another is a look that worked.
- `ScreenLookHealth`: told from the second failed look in a row, once a reason; taken back by the
  first look that works. One failed look is not told (a display that went to sleep under the call).
- A look macOS has not answered in 30 s is given up ("macOS did not answer."), its late callbacks
  dropped by number (`look`), the display list asked for again.
- A segment whose frame could not be encoded or written is closed, so the next shot starts a new
  one with a new encoder and a key frame.
- Shown: Settings > Files > Screenshots, a row "Not recording" with the reason, and a yellow
  triangle on the sidebar row (its tooltip the reason). Turning recording off clears it.

Through the app, isolated (`-omni.screenshots.capture YES`, the real screen, to scratch):
`shotfail:<text>` fails two looks at the next tick: `screen-look failed n=1`, `n=2`,
`screen-recording failing: ...`, the row and the badge, `dumpui` `screenshotsFailure`; the next
real look stored a shot and logged `screen-recording working`, row and badge gone; failing again
and turned off, the failure is cleared. Renders per step (counter build): entering the failing
state 1 `ScreenshotsRow` and 1 of the Settings tab, 20 s in it 0, turning off 1 and 1. This Mac had
seen no input for 9 hours, so the real looks were started with pointer events posted at the
pointer's own place (4, then 7, 2 s apart); nothing was clicked or moved.

A failure that is not scripted: the run's history folder made read-only with recording off, then
on. Three real looks could not be stored (`stored=0`, "The video could not be started."), told at
the second; the folder writable again, the next look stored a shot and took it back. A capture
call failing inside macOS could not be produced on demand; it ends at the same `ended(failure:)`.

## The frame out of the video against the still (measured 2026-11-12)

Han, 2026-11-12: keep the video only, the time as the pointer, the frame taken back out when a
search finds it. Measured before anything is built: is the frame the history gives back as good a
picture for the index as the still? Nothing below is in the app.

Read (`shotbench stillsize`, arms "video q0.7|q0.8 key|delta": the frame encoded as a key frame, or
as the frame after a different screen, decoded and read in pieces as the still is; words read of 4
editor frames):

| desktop | still | no codec | q0.7 key | q0.7 delta | q0.8 key | q0.8 delta |
|---|---|---|---|---|---|---|
| 1512 x 982 @2x | 98.4% | 98.0% | 97.6% | 97.7% | 98.0% | 97.8% |
| 1920 x 1080 @1x | 97.4% | 96.7% | 97.0% | 97.4% | 97.7% | 97.3% |
| 1920 x 1080 @2x | 98.0% | 97.9% | 97.7% | 97.9% | 97.5% | 97.7% |
| 2560 x 1440 @1x | 97.2% | 96.8% | 97.4% | 97.3% | 97.6% | 97.3% |
| 2560 x 1440 @2x | 98.0% | 97.6% | 97.6% | 97.8% | 97.7% | 97.7% |
| 3008 x 1692 @2x | 97.6% | 97.4% | 97.2% | 97.8% | 97.6% | 97.7% |
| 3840 x 2160 @1x | 96.9% | 96.6% | 96.8% | 96.8% | 96.8% | 96.8% |

Within 0.8 points of the still everywhere. Of exact lines the q0.7 delta frame read as many or
more on six desktops of seven (3840 x 2160: 73.6% against 74.8%), the key frame on five. q0.8
reads nothing q0.7 does not.

Embedded (`shotbench frameembed <model> <source> <prose> <photo>`: cosine of the frame's vector,
scaled to the still's size, to the still's; 6 screens a desktop; min / mean):

| desktop | model | q0.7 key | q0.7 delta | still to no codec | next screen |
|---|---|---|---|---|---|
| 1512 x 982 @2x | Nano | 0.991 / 0.997 | 0.996 / 0.998 | 0.989 / 0.996 | 0.670 / 0.692 |
| 1920 x 1080 @1x | Nano | 0.992 / 0.997 | 0.994 / 0.997 | 0.978 / 0.993 | 0.663 / 0.698 |
| 2560 x 1440 @2x | Nano | 0.980 / 0.994 | 0.980 / 0.995 | 0.985 / 0.995 | 0.682 / 0.741 |
| 3840 x 2160 @1x | Nano | 0.978 / 0.988 | 0.975 / 0.986 | 0.987 / 0.991 | 0.719 / 0.770 |
| 1512 x 982 @2x | Small | 0.987 / 0.993 | 0.989 / 0.993 | 0.985 / 0.991 | 0.726 / 0.750 |
| 1920 x 1080 @1x | Small | 0.979 / 0.991 | 0.982 / 0.993 | 0.966 / 0.984 | 0.780 / 0.788 |
| 2560 x 1440 @2x | Small | 0.988 / 0.992 | 0.986 / 0.992 | 0.983 / 0.990 | 0.690 / 0.751 |
| 3840 x 2160 @1x | Small | 0.976 / 0.984 | 0.972 / 0.982 | 0.969 / 0.985 | 0.799 / 0.823 |

The frame is as far from the still as the still is from the picture before any codec, and an
editor scrolled by 70 points is at 0.66-0.82: the same picture to the model. What the stills
cost: the 150-shot session's 47 stills are 5.30 MB beside 3.05 MB of video (63% of its bytes).

## Video only (0.20.3)

Han, 2026-11-12, after the measurement above: keep the video and nothing else. Built so that no
row of the index changes.

- THE PATH IS SPELLED FROM THE SEGMENT. Through 0.20.2 an indexed shot was a flag in its record and
  a picture file, and the index held the file's path. The path is still that one. A segment's
  header carries the offset from UTC its names are spelled at and the side its pictures are read
  at (the reserved word at byte 20: Int16 minutes, UInt16 side; a side of 0 is a segment 0.20.2
  wrote and nobody has taken in yet); a record's flags carry which of the names of one second is
  its own (bits 2-4, with 0x20 for "has one"). Indexed is having a name. The writer starts a new
  segment when the offset or the display's scale changes. A path back to its shot: the name's
  date and time as a wall clock, the segments within 15 hours of it, each one's offset, and the
  shot whose spelled path is the same bytes.
- A CONTENT SOURCE, NOT A FILE (`ScreenContentSource`, as a Photos asset): size, picture, content
  key and "is it still there" are answered by the history. A shot the pass already holds is
  unchanged by definition and keeps its row's date and size. The content key is the shot (time
  and display), not the picture: two shots are never the same frame, the recorder stores a screen
  once. A test feed that repeats frames is therefore new work where 0.20.2 deduplicated it; a
  comparison of writes feeds frames neither index has seen.
- THE UPDATE (`ScreenHistory.migrateStills`), automatic, silent, at launch on a utility queue
  (`AppModel.takeInScreenshotFiles`), and again before the first shot is written if it had failed
  (retried every 60 s). While 0.20.2's folder exists the root is blind: `indexedShots()` is nil,
  the pass leaves every shot's row alone. Per segment, the header first (offset and side read from
  one of its own files: dated at its shot to the millisecond, named in local time, the difference
  is the offset; the picture's long side, if the frame fitted to it is the file's size exactly),
  then each flagged shot takes the number of the file dated at it. A file whose date was lost is
  found by its plain name in a second pass, after every dated file has its shot. A shot whose file
  cannot be told from another's takes a name no file had and is read again from its frame; a shot
  whose file is gone is flagged as any other, and its row leaves at the next pass. When every
  segment is done the folder is renamed to `Screen History/stills.removed` and deleted. The index
  is never written.
- KILLED ANYWHERE, IT ENDS THE SAME. A header is written before the flags read by it, a flag is
  one byte in place, a numbered shot is not looked at again, every file is there until every flag
  is on disk, the folder leaves in one rename, and a leftover `stills.removed` is deleted by the
  next launch.

Measured (M3 Ultra, release, `shotbench migrate | state`; the digest is every segment's header,
every shot's flags and path, and what is left on disk):

| history | picture files | update | steps | killed after step |
|---|---|---|---|---|
| the fed test index, through the app | 82 | 23-24 ms | | |
| one day, 5,760 shots | 1,920 | 153.7 ms | 1,930 | 1, 4, 9, 700, 1928, 1929, 1930 |
| thirty days, 172,800 shots, 240 segments | 57,600 (748 MB beside 171 MB of video) | 4,193 ms | 57,842 | 1, 100, 240, 241, 30000, 57840, 57841, 57842 |

Every killed run, run again, ends at the unkilled digest (9093b3989a82b118 one day,
b4f99c5bf76c290c thirty). The reruns of thirty days take 3.2-3.9 s; killed after 57841 (the
rename), the rerun only finishes the delete, 2.45 s; after 57842, 0.2 ms. After it every frame
the index holds decodes: 57,600 of 57,600, 0 failed, 101 s for all of them.

Through the app, on clones of one index 0.20.2 built from fed shots (321 files, 82 of them shots,
settled, and again with its text backlog unfinished), unkilled and SIGKILLed during the update,
the pass after it and the text backlog: the same state digest (742822a042c82a03) and the same
served answers as 0.20.2 on that index, compared as sets where scores tie (notes/testing.md).
Back to back, three interleaved rounds:

| | 0.20.2 | 0.20.3, first open | 0.20.3, opened again |
|---|---|---|---|
| launch ready | 818-826 ms | 801-830 ms | 808-822 ms |
| 48 served searches, wall | 2,975-3,004 ms | 2,977-2,995 ms | 3,117-3,146 ms (353 files by then) |
| the served work in them, p50 | 4 ms | 4 ms | 5 ms |
| 150 new frames stored | 4.02-4.18 s | | 2.73-2.75 s |
| their 32 indexed shots all searchable | 13.20-13.26 s | | 6.57-6.64 s |
| embedding in that | 3.91-3.92 s | | 3.84-3.89 s |

A stored shot costs less because the HEIC is no longer written beside the frame, and a shot is
searchable sooner because it is handed to the indexer as it is stored (0.20.2 waited for the
watcher to see the file).

## The list of indexed shots

A pass asks which shots the index is to hold. With no folder to list, the answer is a few bytes
in every record of every segment. `shots.catalog` beside the segments keeps each closed
segment's answer with the segment's size and date; a segment that differs, or is new, is walked
again; the open segments are always walked. The catalog is never the authority: deleted or cut
short, the next pass writes it again from the video (ScreenHistoryTests).

| history | walked cold | walked warm | from the catalog | catalog |
|---|---|---|---|---|
| one day, 1,920 indexed | 6.8 ms | 4.5 ms | 1.1 ms | |
| thirty days, 57,600 indexed | 164 ms | 107-113 ms | 23-26 ms | 3.8 MB |

## A shot as a file

Nothing in Omni needs one: a thumbnail, the timeline, `read_file` and a linked Mac's thumbnail
are cut from the frame. Quick Look, Share, Copy, a drag, Open With and a linked Mac's fetch take
a file, so the frame is written out then (`ShotActions.file`, `ScreenHistory.export`): the HEIC
0.20.2 kept, same name, dated at the shot, marked as a screen capture, in a folder of this launch
under the temporary directory (a later launch deletes the folders of launches that have ended).

- One shot: 54.7-64.9 ms the first time (105 KB at 1920 x 1080), 0.2 ms after. A linked Mac's
  fetch of it 42.5 ms, its thumbnail (456 x 257 JPEG, 17 KB) 122.9 ms. The timeline's own drag is
  the whole frame as a PNG: 75 ms, 421 KB. A scrub step p50 23.1 ms, p90 48.3.
- READERS. One decoder walking from the key frame served every read through 0.20.2's timeline; a
  grid of results asks for four thumbnails at once (`Thumbnail.decodeGate`). The history keeps up
  to four readers (`readerCount`), each with its decoder, open segment and last decoded frame,
  and drops to one at rest. 40 thumbnails at 256, three interleaved rounds: p50 10.9-12.5 ms from
  the video against 17.2-19.6 ms from 0.20.2's files; wall 259-319 ms four at a time against
  318-396 ms.
- A READ BESIDE A DELETE. A delete rewrites a segment beside itself and renames it over. A reader
  holds the table it read the offsets from (`tableChanges`), checks each record's head on disk
  before its payload, and a shot from before the delete answers nil instead of another shot's
  frame. Three tests, each failing with its check taken out.
- A DRAG OF MANY. The file is written as the drag starts, on the thread the drag starts on
  (a browser's upload area and a chat window take an existing file, not a promise). Fifteen
  selected shots held it 845 ms, one after another. Up to four are now written together (83.9 ms
  for four against 54.7 ms for one); a larger selection's shots go as promises, written where
  they are dropped as Photos assets are: nothing at the start, fifteen written in 217 ms
  (`FileDrag.shotFiles`, steps `shotfiles` and `shotpromise:<dir>`).
- Delete (a shot, a session, a day): the frames 200 ms, their rows 3 ms. A shot's size in a
  listing is its frame's bytes in the video.
- IN THE MENUS a shot is "Show in Screenshots" (the timeline at that moment; Open goes there too)
  and "Delete…" (confirmed, not undoable); a file is "Show in Finder" and "Move to Trash". A
  selection of both moves its files to the Trash and names their count; the shots stay. Read
  off the File menu through the app (`menustate:File`): one shot, and 4 to 29 of them,
  "Show in Screenshots" and "Delete…"; 25 files with 20 shots, "Show in Finder", "Copy 45 Paths"
  and "Move 25 Items to Trash" (it read 45 before the count was the files'); one file with one
  shot, "Move to Trash", which moved the file, opened no confirmation and left the shot selected.

## A shot as a link (0.20.4)

0.20.3 kept a shot only as a frame of the video, and served it by the path 0.20.2's still had:
`.../Screenshots/2026-11-12/2026-11-12 12.01.23 Browser - Field notes.heic`. Nothing is at that
path, so an agent that opened it with its own reader got "file not found", and the name is full
of spaces. Han, 2026-11-13: a link, dense, no punctuation.

- THE LINK. `omni://screenshots/20261112200123`: the shot's second in UTC as fourteen digits
  (local time repeats in the hour the clocks go back), and `/<display>` only when another
  indexed shot shares that second (`ScreenHistory.link(of:)`). One display takes at most one shot
  in 5 s (`ScreenCadence.interval`), so the second and the display name one shot; a link with no
  display that two shots share reads to neither. The row's `taken` stays in local time with its
  offset, and the app and window in its `screenshot` object. The index is unchanged: its rows keep
  the paths, and the link is spelled at the served surfaces only, so there is no migration.
- WHERE. MCP and the CLI (search, search_inline, read_file, file_status, tag_image), REST rows, a
  linked Mac's answers (`ScreenHints.Moment.link` beside the path its window uses; a Mac older
  than 0.20.4 sends none, and its shots keep their paths), the pool summary's group
  (`omni://screenshots 21`), and the app's Copy Path and the text Command-C puts beside the files.
  Back in, every path argument reads a link to the shot's path (`normalizeStorePath`,
  `LinkResponder.local`); `omni://screenshots` is the folder, for `folder`, `folders`,
  `exclude_folders` and `search_inline`. A link no shot answers says so: "no screenshot at
  omni://screenshots/... in Omni's screen history".
- SAID ONCE. Under results that hold a shot, one line: "A screenshot is no file: read_file with its
  omni:// link returns it as a JPEG." (on a linked Mac's results, get_linked_file with the link; the
  CLI's names `omni read_file`). Not a field on every row: the per-row resource links were 2,079
  bytes of a 9,656-byte answer and were cut for that (index.md). The search and read_file
  descriptions and the server instructions say the same.
- NOT EVERY PICTURE ON THE COMMAND LINE. The plan was a `picture:` file under each shot in every
  CLI search. Measured, 20 searches each, three interleaved rounds against 0.20.3 on the fed test
  index: with ten shots in the answer 6,211-6,576 ms against 665-727 ms, about 285 ms a search
  for ten decodes and JPEGs nobody had asked for; with no shot in it 396-462 against 400-461 ms.
  Dropped: `omni read_file <link>` writes the one picture an agent wants (`$TMPDIR/omni-pictures`,
  per user, cleared at boot; never `/private/tmp`, which every account on the Mac can list), and
  `--include_images` still writes all. With the link and no pictures, 652-792 ms against
  724-831 ms for 0.20.3, and 405-527 against 483-494 for a search with no shot.
- AT SCALE. On a copy of the thirty-day history, updated (57,600 indexed shots, digest
  b4f99c5bf76c290c): 57,600 links, 57,600 distinct, every one read back to its own path, 0.025 ms
  a link p50 (0.052 p99) and 0.023 ms back (`shotbench state --links`).
- A LINKED MAC'S SHOT. Asked for by its link, the copy `get_linked_file` makes is named by it
  (`20261112200143.heic`); asked for by its path, as the window does, by the path's name. Through
  two linked renamed copies: the asking Mac's search lists `omni://screenshots/...` under the
  other's name, `get_linked_file` copies the 125,084-byte HEIC, and `read_file` reads the copy.
- FOUND ON THE WAY, FIXED. (1) A linked Mac's path that is no file (a Photos asset,
  `photos://...`, or a link) was run straight onto the peer's id, `omni-link://abcphotos://...`,
  and the id could not be read back ("abcphotos:"): it is joined with a slash now (`RemotePath`,
  link.md). (2) `tag_image` with `recompute` on a shot said "outside the indexed folders": the
  check looked for the Screenshots folder on disk, which 0.20.3 removed, and past it there was no
  file to decode. A shot the history holds is inside now, and tagging takes its frame (through a
  renamed copy: stored and recomputed tags for a link). `ocr` on a shot had the same check and
  reads the shot's written-out file now (`ShotActions.file`); not run through the app, which
  would load the OCR model. (3) NSString's `expandingTildeInPath` reads `omni://screenshots/x` as a path and
  collapses it to `omni:/screenshots/x`; served paths are expanded only when they have no scheme
  (`expandedPath`). (4) Omni's own copies went to the general pasteboard in an isolated run too:
  with `-omni.clipboard.pasteboard <name>` they go to the named one (`OmniPasteboard.pasteboard`).

## Isolated runs

An isolated run (`-omni.dbDir`) reads `-omni.screenshots.enabled` from its arguments only and
captures the real screen only with `-omni.screenshots.capture YES`; without it it never asks macOS
for the grant. Started from a terminal that may record the screen, it has the terminal's grant
(macOS holds the terminal responsible for what it starts) and records the real desktop at once, no
prompt; started through LaunchServices (`open -n`), it has its own, which a build nobody granted
does not have (`screen-access allowed=false`, the no-access pane). What it records is the user's
screen: it goes to scratch and is deleted with the run. A second Omni on that screen is in the
shots only under another bundle id: ScreenCaptureKit leaves an app out by bundle id, whatever pid
the filter was built from (a 1500 x 950 window turning dark read novelty 0.001; from the renamed
copy of `Scripts/chaos-app.sh`, 0.149 and 0.126, indexed). Otherwise its shots are fed: perf-script `shotfeed:<dir>[|hoursAgo]` puts PNG frames
(`<n>[@<display>][!] <App> - <Title>.png`, "!" for a settled shot, 5 s apart) through the path a capture
takes. Other steps: `screenshots:on|off|clear`, `shotselect:<n>`, `shotscrub:<n>`,
`shotday:<-1|1>`, `shotdisplay:<n>`, `shotzoom:<factor|reset>`, `shotpan:<fraction>`,
`shotdragfile`, `shotdelete:<shot|session|day>` and `shotexclude:<bundle ids>` (both without their
confirmation), `shotdecline:<seconds>` (macOS declining a capture, asked again after that long),
`shotfail:<text>` (two looks failing with that reason), `shotcopy` and `shotopen` (the selected
shot copied, opened), `shotfiles` (the files a drag of the selection writes as it starts, timed
twice), `shotpromise:<dir>` (its promised files written into a folder, as a drop does),
`menustate:<menu>` and `menushow:<menu>` (a menu bar menu's items as text, and open for a picture),
`copypath` and `pasteboard` (Copy Path, and what is on the run's named pasteboard,
`-omni.clipboard.pasteboard <name>`), `sidebarselect:-3`; `dumpui:<path>` carries the feature's state (on, access, the
excluded apps, bytes on disk, the failure told).

## Verification

- Unit (`ScreenHistoryTests`, 39): signature and gate, a shot as a link (UTC, two displays, a deleted neighbour, malformed links), codec round trip, the record format and a
  cut record, segments by hour and launch, a shot's path and the way back from it, the update
  from 0.20.2's files (dates lost, a file gone, two displays in one second, a clock change, killed
  at every step), the catalog (believed when whole; damaged, cut short or gone, written again
  from the video; a segment gone), readers beside a delete,
  sessions (a gap, a long one, several moments at once), retention, clear, a recorder that starts
  again, a deleted span, a history that cannot write as a failure (an unchanged screen is not).
  `ScreenShotsTests` (11): a shot through the indexer with no file, a new shot indexed by its
  path alone, a deleted shot's rows gone by each of the three ways, a cleared history, Images
  off, a forced reading that keeps the words, an index built from 0.20.2's files the same index
  after the update, no row removed while the files wait, shots given new paths read again from
  their frames. `FileFactsTests` (3) and
  `FileKindTests` (2): notes/index.md, "A file's facts".
  `ScreenTextTests` (5): the size read, the pieces, a line and a long word across pieces read once.
  `EnrichmentTests` (8). `OmniCLITests`: an optional bare word. `ScreenCadenceTests` (11): when a
  shot is taken, none with nobody at the Mac, and a failing look told once from the second in a
  row. The suite: 1123 tests, 6 skipped, 0 failures (`LinkNodeTests.testRemotePath` covers the join of a path with a scheme).
- Linked, two isolated instances paired on one Mac: a query on the asking one returns the other's
  shots in the window, over MCP, the CLI and `--json`, each with its moment; the session line
  names the Mac; the range with no query on the named Mac; `get_linked_file` of a shot, and
  `read_file` of that copy as a JPEG (1568 x 882 of a 1920 x 1080 still). In the window: 44 linked
  shots for a query, one fetched in 2.7 ms (91 KB), its thumbnail drawn by the answering Mac
  (456 x 256), and Find Similar on a linked shot: 24 of that Mac's shots, the nearest ones the same
  window seconds apart, the shot itself not among them.
- The real screen (above): capture, the gate after a start, the shot dropped at Turn Off, a
  working session (cadence, the settled shot, an unchanged screen, an excluded app in front and
  behind, bytes), a declined capture and the retry.
- Fed, three displays across midnight, one at 3840 x 2160 (above).
- The timeline's zoom and pan, a shot dragged out as a file, Delete by shot, session and day, the
  apps left out and the space used in Settings: each through the app by its script step, the
  window shown to Han.
- Isolated, through the app: consent screen, Turn On, feed, timeline in light and dark, scrub, a
  search scoped to Screenshots, Settings, Clear (files, rows and the sidebar count gone).
- 0.20.3, through a renamed copy of the final build: the update on an index 0.20.2 built, unkilled
  and killed ("Video only"); search, thumbnails and the timeline over shots with no file behind
  them; the File menu and a row's menu by selection (none, a file, two files, a shot, two shots,
  files with shots: notes/ui.md, "Titles by selection"), Trash on a file beside a shot, the delete
  alert; a drag of one, four and fifteen shots; and a linked Mac asking about a shot that is no
  file on the answering Mac: `similar` 120 hits, 31 of them shots, 10.6 ms; `thumb` a JPEG of
  18,985 bytes, 110.6 ms; `fetch` the HEIC, 96,881 bytes, 51.8 ms. Renders per step counted
  against 0.20.2, the same line for line.
- Served, on the isolated port and socket: a query that finds a shot (time, session line), the
  range with no query (the two files dated inside it; a file dated after it and the shots left
  out), the folder named (shots listed), neither (the error), an empty range, REST both forms,
  `omni search` both forms, `--json`, `--help`; the listing absent from the sidebar history.
