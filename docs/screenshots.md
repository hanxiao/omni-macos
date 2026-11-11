# Screenshots (2026-11-10)

Omni takes a shot of the desktop while the Mac is in use, keeps every shot as video, and writes
the few worth finding again as picture files into a folder the index already crawls. Search,
filters, thumbnails, Quick Look, Find Similar, MCP, CLI and REST are the existing pipeline,
unchanged: a shot in the index is a screenshot file like any other (`Indexer.isScreenshot`, a
picture embedding plus `ScreenText`). No schema change, no migration, no new model, no new package.

The one thing served surfaces add is time: a screenshot result says when it was taken and what the
stretch of work around it was, and `search` lists the files changed in a time range with no query.
That is the chain an agent follows: query -> a screenshot -> its time -> the files changed then.

## Decisions

- Opt-in, like the clipboard. Off until turned on from the sidebar row or Settings > Files; macOS
  then asks for Screen & System Audio Recording, which only the user can grant.
- A shot follows input, not a clock (`ScreenRecorder`): one every 5 s while there is input (keys,
  pointer, scroll: `CGEventSource.secondsSinceLastEventType`, no permission needed), one more when
  input stops (marked settled), then a look at 30 s that doubles to 5 min while the screen stays as
  it was. A screen that moves by itself (a build, a video) keeps the 30 s.
- No input and a still screen writes nothing: a shot whose signature equals the last stored one is
  not stored (`ScreenHistory.record`).
- Every stored shot is in the timeline; few are in the index (`ScreenGate`). A shot is indexed
  when at least 10% of its tiles both changed and show something, against the nearest of the last
  8 indexed shots; a settled shot at 0.5%. A flat screen is never indexed.
- Not taken: screen locked, another user at the console, screen saver, display asleep, a password
  field holding the keyboard (secure input). Omni's own windows are left out of the shot (a search
  in Omni would otherwise be found by the picture of its own results). The pointer is not drawn.
  Windows an app marks as not shareable are left out by the system.
- Retention 30 days by default; 7 / 30 / 90 days / forever in Settings, with the space used
  (video and stills). Clear is on the sidebar row's context menu.
- Apps left out (Settings > Files, "Apps left out of screenshots"): their windows are not in a
  shot, and no shot is taken while one of them is in front.
- Served to this Mac's agents (MCP, CLI, REST, Shortcuts) as the user's files are, and to Linked
  Macs as the rest of the index is (Han, 2026-11-10; the first build kept them back with the
  clipboard, which alone stays in `AppModel.unsharedFolders`).

## Parts

1. `Sources/OmniKit/ScreenChange.swift`: `ScreenSignature` (green channel on a grid about 960
   wide, tiles of 16 x 16 samples, which tiles show something) and `ScreenGate`.
2. `Sources/OmniKit/ScreenCodec.swift`: VideoToolbox HEVC on the hardware encoder, quality 0.7, a
   key frame every 30 shots, no frame reordering; the decoder; the still writer.
3. `Sources/OmniKit/ScreenHistory.swift`: two folders beside the index.
   `Application Support/Omni/Screen History` holds the segments (`<start>-<display>.omniscreen`,
   one per display per hour or 720 shots, a new one at every launch): records appended one at a
   time, each with its time, the app in front, its window's title and the HEVC frame. Never
   crawled. A record cut short by a kill is the last thing in its file and a reader stops at it:
   nothing to repair. The frame table is read from the segments; there is no second file.
   `Application Support/Omni/Screenshots` is the crawl root: an indexed shot is also written there
   as `2026-11-10/2026-11-10 13.16.27 Editor - source.swift.heic`, a pixel a point of the display
   (`ScreenHistory.stillSide`; 0.20.0 wrote at most 1568 wide, which a desktop wider than that
   could not be read from: "The reader"), dated at the shot and marked `kMDItemIsScreenCapture`.
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
   confirmation that names the count: the stills first, then the frames (`ScreenHistory.delete`:
   the frames after a deleted one, up to the next key frame, are encoded again from a new key
   frame, the segment written beside itself and renamed over).
6. `App/Serving/ScreenHints.swift`: what a served screenshot result says (below).
7. AppModel: the folder joins `crawlRoots` when it exists, outside `addedFolders`, exempt from the
   ignore rules as the clipboard's is.

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
session's own mix, pauses included, is 15.0 MB an hour: 3.6 GB. Settings shows the space really
used.

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
`sidebarselect:-3`; `dumpui:<path>` carries the feature's state (on, access, the
excluded apps, bytes on disk).

## Verification

- Unit (`ScreenHistoryTests`, 23): signature and gate, codec round trip, the record format and a
  cut record, segments by hour and launch, stills and their dates, sessions (a gap, a long one,
  several moments at once), retention, clear, a recorder that starts again, a deleted span, the
  whole frame of a still. `ScreenTextTests` (5): the size read, the pieces, a line and a long word
  across pieces read once. `EnrichmentTests` (8). `OmniCLITests`: an optional bare word. The suite:
  1080 tests, 6 skipped, 0 failures.
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
- Served, on the isolated port and socket: a query that finds a shot (time, session line), the
  range with no query (the two files dated inside it; a file dated after it and the shots left
  out), the folder named (shots listed), neither (the error), an empty range, REST both forms,
  `omni search` both forms, `--json`, `--help`; the listing absent from the sidebar history.
