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
- Retention 30 days by default; 7 / 30 / 90 days / forever in Settings. Clear is on the sidebar
  row's context menu.
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
   as `2026-11-10/2026-11-10 13.16.27 Editor - source.swift.heic`, at most 1568 on its long side
   (what the reader reads at), dated at the shot and marked `kMDItemIsScreenCapture`.
4. `App/ScreenRecorder.swift`: one `SCScreenshotManager` still per look and display (not a stream:
   a stream delivers frames whether or not anything moved), on its own queue.
5. `App/ScreenTimeline.swift`: the Screenshots place. One day on a track true to time, a point a
   shot (indexed ones larger), a playhead to drag, arrow keys, a day picker, the shot's time, app
   and window. A decoded shot goes straight into a layer, so a drag draws frames with no SwiftUI
   render. Off and empty it is the standard empty state with Turn On; denied, a button to Settings.
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
- Signature 0.38 ms a frame, compare 0.33 ms a pair. The reader at 1568 wide: 144 ms a frame.

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
3.9 MB, about 83 KB each, so the indexed pictures take more disk than the whole video. A scrub of
349 steps: 347 decodes, p50 2.6 ms, p90 22.2 ms, max 54.0 ms.

Served:

- The hint (`shotbench hints <dir> 30`: 30 days, a shot every 5 s for 8 h unbroken, 172,800
  shots, 240 segments): the stretch around a shot 0.79 ms warm (p90 0.86), 3.1 ms the first time
  a day's segments are read (max 9.7). Thirty screenshot results walked one by one were 22.6 ms;
  a stretch is walked once however many results fall in it (`sessions(around:)`): 2.5 ms for
  thirty over three days.
- The range listing (`shotbench listrange <clone>/index.sqlite`, the 9.73M-row bench index):
  162 ms p50 whatever the range is (13 min, an hour, a day, 30 days, top 30 or 190), against 183
  ms for the same listing with no filter. It is a walk of every resident row on the store queue,
  the hold the app's own tag browsing already takes. OPEN: a search that arrives during one waits
  it out. A walk by file (2.7M) instead of by row needs a change in the store and was not made.

## The still that cost the video its reference

ImageIO's HEIC writer on the hardware encoder starts an HEVC session per picture, and the video
session's next frame came out a key frame: every indexed shot was followed by one, and the session
grew to the size of its stills. `ScreenCodec.stillOptions` asks for the software encoder
(`kCGImageDestinationUseHardwareAcceleration: false`). At 1568 x 882, q0.8: hardware 48,613 bytes
and 40-45 ms (84 the first); software 49,802 bytes and 33 ms.
`testTheShotAfterAnIndexedOneIsStillSmall` holds it.

Why HEIC: the six real stills (1568 wide, q0.8) are 71-86 KB; the same pictures as JPEG q0.8 are
152-164 KB (1.9x) and as PNG 307-319 KB (3.8x), re-encoded from the stills with `sips`. The stills
are about half of what the feature stores. The cost: a HEIC is a path some agents cannot open; MCP
`search` inlines image hits as JPEG thumbnails (800 px), the CLI gives the path alone.

## A real session (2026-11-10)

An isolated development build with `-omni.screenshots.capture YES` on Han's Mac: one display,
1920 x 1080, the installed app's grant (no prompt). Han was away, so only the idle look ran on real
content; a shot driven by input, and a settled one, have not run on a real screen.

- Capture 46-111 ms. Record 8-11 ms for a delta, 87-280 ms when a segment starts or a still is
  written. A key frame 72-88 KB, an idle look's delta 0.6-3.4 KB at novelty 0.005-0.009 (stored,
  not indexed), a still 71-86 KB.
- Footprint with the engine quiet for 70 s: 2017 MB with recording off, 2021 MB on. Timeline open,
  a 65-step scrub, closed: within 9 MB. An indexed shot is an image through the engine, which
  holds about 585 MB more for about 60 s after any image.
- Scrub on real shots: 63 decodes, p50 3.7 ms, p90 11.9, max 16.9.
- A search in the window with no scope returns a real shot first among ordinary files.

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

Thirty days, estimated and not measured: 400-720 shots an hour of use at 16-48 KB a shot (video
plus stills; the low end typing and reading with a tenth indexed, the high end the drawn session
unbroken) is 6-35 MB an hour, 1.5-8.4 GB for 240 hours at 1920 x 1080. The cadence of real work is
the assumption.

## Isolated runs

An isolated run (`-omni.dbDir`) reads `-omni.screenshots.enabled` from its arguments only and
captures the real screen only with `-omni.screenshots.capture YES`; without it it never asks macOS
for the grant. A development build run isolated has the installed app's bundle id and so its grant:
with the flag it records the real desktop at once, no prompt. What it records is the user's screen:
it goes to scratch and is deleted with the run. Otherwise its shots are fed: perf-script `shotfeed:<dir>[|hoursAgo]` puts PNG frames
(`<n>[!] <App> - <Title>.png`, "!" for a settled shot, 5 s apart) through the path a capture
takes. Other steps: `screenshots:on|off|clear`, `shotselect:<n>`, `shotscrub:<n>`,
`shotday:<-1|1>`, `sidebarselect:-3`.

## Verification

- Unit (`ScreenHistoryTests`, 18): signature and gate, codec round trip, the record format and a
  cut record, segments by hour and launch, stills and their dates, sessions (a gap, a long one,
  several moments at once), retention, clear, a recorder that starts again. `OmniCLITests`: an
  optional bare word. The suite: 1067 tests, 6 skipped, 0 failures.
- Linked, two isolated instances paired on one Mac: a query on the asking one returns the other's
  shots in the window, over MCP, the CLI and `--json`, each with its moment; the session line
  names the Mac; the range with no query on the named Mac; `get_linked_file` of a shot. Find
  Similar across the link was not run.
- The real screen (above): capture, the gate after a start, the shot dropped at Turn Off.
- Isolated, through the app: consent screen, Turn On, feed, timeline in light and dark, scrub, a
  search scoped to Screenshots, Settings, Clear (files, rows and the sidebar count gone).
- Served, on the isolated port and socket: a query that finds a shot (time, session line), the
  range with no query (the two files dated inside it; a file dated after it and the shots left
  out), the folder named (shots listed), neither (the error), an empty range, REST both forms,
  `omni search` both forms, `--json`, `--help`; the listing absent from the sidebar history.

## Not measured, not built

- A working session. The real run above had nobody at the Mac: the cadence under input, the gate
  on real work and disk use per day are still to be read. A shot every 5 s for 8 h is 5,760 shots.
- The system's recording indicator and its recurring prompt, more than one display, and displays
  above 4K, where a still read at 1568 wide may lose small text.
- A picture an agent can open from the CLI: the still is a HEIC, and nothing converts it on request.
- Deleting a span of shots, a list of apps never to record, zooming the track, dragging a shot
  out, a disk-use line in Settings (`ScreenHistory.diskUse` exists).
