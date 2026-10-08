# Testing, chaos runs and UI responsiveness

Moved out of CLAUDE.md on 2026-10-03, section text unchanged except where marked SUPERSEDED or
UPDATED. Dated findings, measurements and rejected options: read the section before touching
the code it names.

## Chaos testing the UI (2026-09-11, Tools: /tmp/chaos/drive.py pattern)

A seeded randomised driver (PyObjC `CGEvent`) against a `-omni.hangwatch YES` build. Destructive
actions are blacklisted deliberately: no Cmd-Delete (trashes files), no Cmd-D, no Cmd-O /
Cmd-Shift-O (a modal file panel blocks the event tap and wedges the run), no right-clicks (the
context menus carry "Remove from Omni"), no Settings. Seed the RNG so a finding replays.

FOUND AND FIXED - a hard crash, `OCRBatched.swift` `decodeContinuous`:
`Fatal error: Unexpectedly found nil while implicitly unwrapping an Optional value`, stack
`closure #8 in decodeContinuous` inside `Collection.map`. Admission is lazy - `nextPage` walks
forward one row per decode step - so an early exit (Cmd-. through `shouldContinue`, or every row
vacating) leaves `[nextPage, n)` never prepared. The trailing `map` then ran over ALL n pages and
force-unwrapped `pages[p].prep.grid` on a page that has no `prep`. Exactly the hazard the comment
on `ensurePrepared` already warns about, 190 lines further down. Two parts to the fix: `prep?.grid
?? (0, 0)`, and marking those pages `.cancelled` - `stopped` defaults to `.cap`, so a page that
never started was reporting as one that ran to the token cap. Reproduced with seed 4242, verified
fixed on the same seed and on seed 777 with two documents.

FOUND, NOT FIXED - AppKit logs "Application performed a reentrant operation in its NSTableView
delegate. This warning will become an assert in the future." Deterministic on seed 1337 in SEARCH
mode: clean at 110 and 165 actions, fires at 188 and 220 (confirmed on four separate clean
processes). The site is NOT isolated. Three hypotheses were tested and all disproved: rapid
history-row selection, a Quick Look storm, and a back/forward storm with a sidebar row selected.
Deferring the three `selection = nil` writes in `Sidebar.swift` out of their `onChange` handlers
did NOT silence it either, so that change was reverted rather than left in as a speculative fix.
Two traps for whoever picks this up: AppKit logs the warning ONCE PER PROCESS, so any repro attempt
needs a fresh launch or it silently proves nothing; and it does not reproduce under lldb at all
(it is timing-dependent), while a breakpoint on `NSLog` never fires because the warning goes
through `os_log`.

Main-thread stalls over 250 ms, with the index live: 5-6 per 220 search actions, worst 396-440 ms;
3 per 200 OCR actions. Consistent with the known remaining SwiftUI/CoreText cost. SUPERSEDED: a
day later the same tour measured ZERO blocks over 250 ms (see "UI tests").

## UI tests (UITests/, Scripts/ui-test.sh)
- `-omni.hangwatch YES` DID NOTHING FOR AN UNKNOWN STRETCH, and the stall figures quoted in this
  file came from it. It was started from a `.task` on the window's content that never ran: a probe
  as the first statement of that task - writing unconditionally to stderr, then to a file - produced
  nothing across repeated launches, while sibling `.task` modifiers in the SAME chain
  (`-omni.ocrOpen`, `-omni.query`) fire every time. The break predates 2026-09-12: `9b1e589` has the
  identical `.task { ocrOpen } / .frame / .task { hangwatch }` shape. It now starts from
  `applicationDidFinishLaunching`, which is guaranteed to run, and takes `-omni.hangwatchFile
  <path>` because under XCUITest the app's stderr goes into the test bundle and cannot be read from
  a shell.
- ALWAYS RUN THE POSITIVE CONTROL. A detector reporting nothing looks exactly like one that is not
  firing - which is how "0 stalls" was nearly reported off the dead instrument. Drop the threshold
  (`-omni.hangwatchMs 30`) and check it produces hundreds of lines before believing a zero at 250.
- MEASURED 2026-09-12, cross-view tour under index churn (sidebar, browse roots, gallery/list,
  enclosing folder, back/forward, search, OCR in and out, Settings), and again inside a full chaos
  run: ZERO main-thread blocks over 250 ms. Positive control at 30 ms on the same build: 457 blocks,
  worst 179 ms. So the app does not stall; the driver failures below are the harness.
- THE CHAOS SUITE IS FLAKY AND IT IS NOT A REGRESSION. It fails perhaps half the time with "Failed
  to synthesize event: Timed out while synthesizing event", "window was gone at the end", or
  "Application has not loaded accessibility" on a back-to-back launch. Checked rather than assumed:
  a worktree at `9b1e589` - before any of the 2026-09-12 work - fails the same suite with the same
  "window was gone at the end". Re-run before believing a red, and capture the error line.
- CMD-W IS AMBIGUOUS IN A CHAOS LOOP. It closes the FRONT window, so a Settings round whose
  Cmd-comma did not land hides the MAIN window instead, every later round drives nothing, and the
  run ends blaming the app. Close conditionally on a second window actually existing.
- `activate()` IS NOT A REOPEN. Closing the window hides it (close-to-hide), and
  `applicationShouldHandleReopen` fires for a Dock click, Spotlight and `open -a` - not for
  Cmd-Tab or `XCUIApplication.activate()`. That is the same as Chrome with its last window closed.
  A test asserting the window returns after `activate()` is testing a path no user takes.
- THE HANDOFF TESTS TAKE THEIR QUERY FROM A LAUNCH SEAM (`-omni.query`), not from typing. XCUITest
  could not get text into the toolbar's `.searchable` field in that suite - exists / enabled /
  hittable all true, `typeText` lands nowhere - so all three selection tests SKIPPED for a session
  while looking like coverage. The seam goes through `applyParsedQuery`, the same door a typed
  query uses, so chips, qualifier bar and store filter are built exactly as they would be.
- CHAOS DOES TYPE, AND THIS WAS CHECKED RATHER THAN ASSUMED. The suspicion that `ChaosUITests` had
  never typed either (it only asserted the app was alive) is WRONG: with an assertion on the
  field's value it reports `field="porsche"` and passes. Anything that suite says about search is
  therefore about search.
- BOTH UI SUITES FLAKE, at the XCUITest level rather than the app's. Chaos fails ~1 run in 2 with
  "Failed to synthesize event: Timed out while synthesizing event" mid-loop; the handoff suite was
  seen failing once in three full-suite runs while passing in isolation and on the next full run.
  Re-run before believing a red, and CAPTURE THE ERROR LINE when you do - a filtered grep that
  keeps only "Test Case ... failed" throws away the one piece of evidence that would say whether it
  was the synthesizer or the app.
- ASSERT THE SELECTION COUNT, NOT THAT THE WORKSPACE OPENED. `Transcribe.title` renders
  "Transcribe 3 Items", so the File menu item's title is a readable statement of what the app
  thinks is selected. Asserting only "did OCR mode open" passes with ONE row selected, which is how
  a three-row test would have gone green while selecting one.
- THE RESULTS LIST HAS NO ARROW-KEY SELECTION. It is a ScrollView of custom rows, not a `List`; its
  only key handler is Return. Click selects, Cmd-click toggles, Shift-click extends - Finder's
  modifiers, but no keyboard range. A test using shift-down to build a run selects nothing.
- A MODIFIED CLICK IS `XCUIElement.perform(withKeyModifiers:)`, and it is main-actor isolated, so
  the test method needs `@MainActor` (not the whole class - that makes `setUpWithError` fight the
  isolated stored properties). A chord sent straight after one intermittently times out in the
  event synthesizer; settle ~0.7 s first.
- `Scripts/ui-test.sh` PROBED THE LEGACY OCR PATH (`Omni/ocr/*`) long after the weights moved to
  `Omni/jina-ocr-v1-<slug>`, so it printed "OCR model not installed" on every machine that had it.
- XCUITest, not synthetic events. `./Scripts/ui-test.sh [Class[/test]]` - never a bare
  `xcodebuild test`, same SE-0482 tokenizers artifact reason as Scripts/build-app.sh.
- CGEvent key PRESSES from an external tool (cliclick `kp:`) never reach the app; typed TEXT does.
  A harness built on them proves nothing about Space/Return/Escape - that is why these exist.
- The runner is SANDBOXED: `.applicationSupportDirectory` inside it resolves to
  `~/Library/Containers/io.hanxiao.omni.uitests.xctrunner/Data/...`, so a filesystem check made in
  a test looks in the wrong place and skips every run while appearing to pass. Ask the APP (it
  draws `ocr.needsmodel` when the optional model is absent).
- `.accessibilityIdentifier` on a container OVERRIDES its children. An id on the tab strip made
  every tab answer to the strip's id and none to its own.
- `-omni.forceOnboarding YES` puts the app on the first-run screen on a machine that already has
  the model. A fresh `HOME` does NOT work for this: `.applicationSupportDirectory` resolves from
  the user record, not the environment, so the app finds the real install and the real defaults.
- `-omni.ocrOpen <path>[:<path>]` opens documents in the OCR workspace; launch arguments land in
  NSUserDefaults' ARGUMENT domain, so a run cannot touch the real index, roots or settings.
- "Timed out while enabling automation mode" = a stale OmniUITests-Runner, the display asleep, or
  - most often - THE PASSWORD PROMPT WENT UNANSWERED. The three look identical from the log. Tell
  them apart by SCREENSHOTTING while the runner waits: the dialog reads "XCTest is trying to Enable
  UI Automation. Enter the password for the user ...". The runner gives up 60 SECONDS after putting
  it up, so there is no way to wait it out and `Scripts/automation-window.sh` cannot be made to
  wait longer - the holder is already dead by then. An unattended UI run therefore has to be
  STARTED while somebody can type, and a job queued after the window closes fails on the harness,
  not on the app.
- THE WINDOW HAS NO TEN-HOUR CAP. That was written here after a 600-minute hold expired, and 600
  was just the number passed. The deadline is the script's argument: 1500 minutes holds 1500. To
  cover a whole day from one password, hold it with a command that outlives the work and detach:
  `nohup ./Scripts/automation-window.sh 1500 sleep 86400 >/tmp/omni-auto.log 2>&1 & disown` - then
  every later `xcodebuild test` finds `/var/db/com.apple.dt.automationmode/automation-enabled`
  already there and never re-authenticates. `touch /tmp/omni-automation-window.release` ends it.

## UI responsiveness (App/OmniApp.swift HangWatch)
- APP NAP (2026-10-03). An isolated instance launched from the shell is a background app, and ~30 s
  in macOS naps it: the main thread moves to efficiency cores (747 P / 164 E samples before, 5 P /
  6,468 E after, Time Profiler's core column) and every UI step costs 4-5x - a search 125 -> 650 ms
  with nothing in the app changing, which read as "the 9th search is slow", then as "it gets slow
  after 30 s". `-NSAppSleepDisabled YES` on the command line keeps it awake (argument domain, so the
  installed app's prefs are untouched): 16 searches at ~125 ms flat. It also explains numbers that
  "drifted" across a sitting - the same HEAD binary read 2.8 s and 13 s of OCR-run stalls on one
  afternoon. Interleaved A/Bs stay valid within one regime; absolute numbers need the flag.
- `-omni.hangwatch YES [-omni.hangwatchMs N]` reports how long the MAIN THREAD was unresponsive.
  A timer on the main run loop only fires when the thread is free, so the gap between firings is
  the block. This is how "feels laggy" becomes a number; N defaults to 250 ms.
- DRIVE THE UI FROM THE LIVE WINDOW FRAME, never hardcoded screen coordinates. The window moves
  between launches: a whole round of measurements was taken with the toolbar clicks landing at
  y=155 while the window started at y=249, so every "interaction" phase measured an app that was
  never clicked, and read as 0 stalls. Get it from System Events
  (`get {position, size} of window 1`) each run and add offsets.
- And RAISE the window: an occluded window does far less work, so decode-phase stall counts are
  not comparable between a raised and an unraised run. An early "79 stalls -> 1" claim here was
  that confound; the controlled A/B with the window visible in both is 2 stalls -> 1.
- THE SHARE ITEM REBUILT THE WHOLE TRANSCRIPT. `TranscriptFile(markdown:)` took a String, so the
  toolbar concatenated every page each time it was rebuilt. It takes a closure now, so the
  document is only produced when a share actually happens. Found by sampling, not by guessing.
- THE VIEW-MODE PICKER NEEDS A FIXED WIDTH. An `NSSegmentedControl` recomputes
  `intrinsicContentSize` through the constraint system on every toolbar re-layout, and the
  toolbar re-lays out whenever any item changes - including the share item, whose title is the
  visible document's name. That was 10% of main-thread samples, and 41 stalls over 18 tab
  switches. `.frame(width: 132).fixedSize()` takes it to 0. Do not remove the frame.
- THE THREE INTERACTIONS IN THE OLD NOTE NO LONGER STALL. Measured on the current build with the
  run allowed to FINISH first, so decode is not in the numbers: 18 view-mode switches, 10
  thumbnail clicks and 4 OCR toggles add ZERO stalls over 120 ms (39 before, 39 after each
  phase). The earlier "about two stalls each at a median of 184 ms" is not reproducible - the
  lazy sections and the pinned picker width appear to have taken it, or it carried the
  duplicate-instance confound below.
- A LOCKED SCREEN LOOKS EXACTLY LIKE A BROKEN HARNESS. `loginwindow` becomes the frontmost
  process, the window query returns nothing and every click is refused; `caffeinate -d -i -s`
  stops sleep but not the lock. Check `first process whose frontmost is true` before concluding
  the app is at fault - twice here the answer was that the machine had simply locked.
- ASSERT WHICH APP OWNS THE FRONT WINDOW BEFORE EVERY CLICK, not once at the start. Focus is
  stolen mid-run (Slack did it here during a 95 s wait) and the clicks then land in another
  application, which measures nothing and does something unintended. `raise` by unix id, then
  check `unix id of first process whose frontmost is true` equals the pid, and ABORT on a
  mismatch. Verified to discriminate: 1603 -> our pid -> 1603 across a raise and a steal.
- WHAT REMAINS IS DECODE-PHASE AND MODEST. On the 40-page scan, one instance: ~40 stalls of
  120-200 ms, ~6.5 s of a ~57 s run. NONE exceed 250 ms, which is the threshold the earlier
  "decode 1 stall" note used - so that figure and this one agree, they were counting different
  things. Fitting blocked time against page count gives ~123 ms per page plus ~1.7 s fixed.
- THE CAUSE WAS `pages[index].tokens`, WRITTEN ON EVERY STREAM UPDATE. `Page.tokens` is read in
  exactly one place, `case .done: "\(page.tokens) tokens"`, and `settle` sets it from the final
  result - so the per-update write displayed nothing and mutated the whole `pages` array 24 times
  a second, invalidating all 40 rows of the rail. `PageThumb.body.getter` is what gave it away in
  the Instruments profile. Removing it from both stream handlers, measured with focus HELD in
  both arms and zero steals: 38/43 stalls and 5.79/6.49 s blocked before, 22/21 stalls and
  3.09/2.87 s after - 51% less main-thread blocking for two deleted lines.
- HOW IT WAS FOUND, after seven guesses failed: `xcrun xctrace record --template 'CPU Profiler'
  --attach <pid>`, then export the `cpu-profile` table and aggregate frames for the Main Thread
  row only. `sample` wedges on this binary; xctrace does not. The profile also settles two
  questions: SQLite and the ignore matcher run during an OCR run but on WORKER threads, so
  indexing is not what stalls the UI; and Instruments' own hang detector records ZERO hangs,
  because its threshold is 250 ms and nothing here exceeds that.
- WHAT THE MAIN THREAD SPENDS THE REST ON, from that profile: SwiftUI environment and attribute
  graph traversal (`find1` 562 samples, `Attribute.init` 293, `PropertyList.Tracker.value` 128,
  `EnvironmentBox.update` 78) plus AttributedString-to-NSAttributedString conversion and CoreText
  layout (`transformingAppKitAttributedForSwiftUI` 92, `NSCoreTypesetter` and `TCompositionEngine`
  147, `Text.Style.nsAttributes` 68). That is the cost of re-rendering visible transcript
  sections, and it is inherent to drawing the text.
- FIVE CANDIDATES FOR THAT PER-PAGE COST WERE MEASURED AND REJECTED BEFORE THE REAL ONE, do not
  re-try them:
  coalescing the 768 per-row main-actor hops a second into 24; publishing only the focused page
  at full rate; memoising `MarkdownBlock.runs` the way `parse` is (worth ~5%, inside variance);
  guarding the run-follow `scrollTo` so it fires once per page rather than every tick; and
  lowering the stream flush rate (worth ~15% of total blocked, not the 45% the percentage
  metric implied). All four code changes were reverted rather than shipped unproven.
- NEVER `tell application "Omni" to activate` IN A HARNESS. It resolves by NAME through
  LaunchServices and launches /Applications/Omni.app as a SECOND INSTANCE, so the build under
  test then competes with a whole other copy of the app for the GPU and memory. This inflated a
  whole afternoon of stall numbers: with the duplicate running a 40-page run reads 91% of the
  decode window blocked with 1000 ms maxima; with exactly one instance the same build reads
  ~40% and 207 ms. Activate by unix id instead -
  `set frontmost of (first process whose unix id is <pid>) to true` - and assert
  `pgrep -x Omni | wc -l` is 1 before believing anything.
- MEASURE TOTAL BLOCKED MILLISECONDS, NOT THE PERCENTAGE. The stall window's span depends on when
  the run happens to start and finish inside a fixed observation window, and model load varies by
  ~13 s run to run, so the same build reads 16% or 53% purely from where the run landed. Total
  blocked time is stable: ~6.8-7.4 s in every configuration tried.
- WHAT THE DECODE-PHASE STALLS ARE NOT. On the 40-page scan with one instance: ~45 stalls of
  ~150-200 ms, ~7 s total, and an IDLE app takes ZERO. Ruled out by measurement, each with a
  reproducible A/B: window visibility (hidden 51.3% against visible 50.6% - hiding a window does
  not stop SwiftUI evaluating bodies), batch width (1 / 8 / 32 all land at the same total blocked),
  main-actor hop count (coalescing 768 per-row hops a second into 24 changed nothing), and the
  stream flush rate (24 Hz against 4 Hz is worth ~15% of the total, not the 45% the percentage
  metric suggested). Both speculative fixes were reverted rather than shipped unproven.
- (Resolved 2026-10-03, below.) WHAT IS LEFT POINTING AT. ~40 pages, ~45 stalls, ~150 ms each: the shape says per-PAGE
  completion, not per-token streaming. `settle` itself is a handful of assignments, so the cost is
  the SwiftUI update its `state = .done` and final text assignment provoke, once per page. That is
  where to look next; do not re-litigate the four above.
- FIXED 2026-10-03: THE PER-PAGE STALL, AND WHAT IT WAS. Each finished or followed page cost
  ~150-350 ms of main thread on a quiet machine, and 1-2 s per jump once the window was in front
  and the GPU busy. Four causes, each found by counting what ran inside a stall (`UIProbe.count` in
  the bodies, hangwatch prints the tallies per stall) and `Self._printChanges()`, not by profile
  shape alone - the profile only ever said "layout and CoreAnimation commit":
  - A FINISHED SOURCE PAGE WAS ONE SwiftUI `Text`. Offscreen harness, the 151-line table page, three
    sections the way a follow jump builds them: one `Text` each ~760 ms, one `Text` per line ~170,
    an `NSTextView` (TextKit 1) ~40. `textSelection` itself costs nothing there; long attributed
    text in SwiftUI does. Source sections are `SourceSection` now (non-editable `OCRSourceTextView`,
    sized in `sizeThatFits` from its layout manager, height cached per width); a streaming page is
    one for its settled lines plus one `Text` for the line being written, and the same text view
    takes the tail when the page finishes. Pixel diff against the `Text` rendering: 32 of 1.26 M
    pixels, glyph edges. TRAPS: a text system built by hand loses an unretained `NSTextStorage` and
    every section sizes to zero (verify by screenshot, not by stall counts - zero-height sections
    are fast); `sizeThatFits` is asked at an infinite width for the ideal size.
  - EVERY SECTION AND THUMBNAIL OBSERVED EVERY PAGE. Observation tracks a stored property whole, so
    `texts[id]` / `pages[id].state` made each built section, the section list and every thumbnail's
    drag payload depend on every token of every page and every thumbnail landing. Per-page
    `SectionSource` objects mirror text and state from `didSet` and the views read those.
  - THE RAIL RE-RAN ALL 40 ROWS per `pages` write: `PageThumb` took a closure and read
    `sourceURL(for:)` (all of `pages`). Equatable now, no closure, URL from its own page: ~4,500
    thumbnail bodies a run -> ~100.
  - THE WINDOW RE-RAN PER PAGE: `ContentView.ocrDrawerWanted` read `pages.isEmpty`, the OCR toolbar
    and the File menu read `pages.isEmpty` and `completedPages == 0`. Stored `hasPages` /
    `hasCompletedPages`, written when they flip: `OCRView` re-creations 229 -> 6 a run.
  Measured, 40-page scan, window visible, interleaved with HEAD (4fcd539) built the same way:
  Raw (the default) 11.8-14.5 s blocked, worst 2.2 s -> 3.0-3.3 s, worst 0.75-0.98; Markdown
  11.7-12.7 s, worst 1.4 s -> 7.3-8.4 s, worst 0.75-1.0; Dual 21.4-21.7 s, worst 3.4-3.5 s ->
  9.2-9.5 s, worst 0.85-1.0. Twelve page jumps in Dual on a finished transcript: 6.1-6.4 s of the
  session blocked -> 1.7-1.8 s. Same tok/s. The SAME HEAD binary read 2.8 s earlier in the day: an
  occluded window does far less work, so the absolute numbers only compare within one sitting.
  WITHOUT APP NAP (see "APP NAP" above), the numbers that describe a window in front - Raw 2.6-2.8 s
  blocked, worst ~415 ms -> 0.3 s, worst 140-163; Markdown 2.2 s, worst 230-314 -> 0.7-0.8 s,
  worst ~155; Dual 3.9-5.0 s, worst 440-701 -> 0.9-1.1 s, worst ~165. Nothing over ~170 ms in any
  mode, 540 tok/s in both arms. The napped figures above overstate every cost 4-5x.
  MEASURED AND NOT DONE: the readout's numeric-roll transitions (no change), the stream fade
  animation (no change), baseline vs leading grid alignment in tables (3.3 vs 3.1 ms a streamed
  update - tables stream cheaply in isolation). The Markdown pane's remaining cost is SwiftUI
  selectable text: with selection off pane-wide a run blocks ~5.2 s against ~8.4 s; finished pages
  stay selectable, the page being written no longer is (8.6-8.9 s against 10.2-11.0 s, napped).
  Prose and tables in TextKit (NSTextTable) would go further, but un-napped the pane already peaks
  at ~155 ms, so it is not worth the visual risk.
- Where it stands, measured with verified coordinates on an 11-document chaotic drop: decode 1
  stall, tab switches 0, and mode switches / thumbnail clicks / OCR toggles about two stalls each
  at a median of 184 ms, p90 242, max 296. Perceptible stutter, no freezes, nothing over 300 ms.

## Search and browse responsiveness (2026-09-23, Scripts/perf-tour.sh)
- THE TOOL: `Scripts/perf-tour.sh <corpus|-> <index> <out> [reserve]` drives
  `UITests/PerfTourUITests` (typing, gallery, list, selection, Find Similar, folder browser) in a
  RELEASE build (`OMNI_UI_CONFIG=Release`; Debug stalls are not the shipped app's) and records the
  stall log, the perf log and phys_footprint; `Scripts/perf-tour-analyze.py <out>` buckets them by phase. `-`
  tours the app's own folders; it then snapshots and restores the user's prefs around the run.
- FOR A REAL INDEX: APFS-clone it (`cp -c -R`, instant) with the app quit, and run a copy of the
  build re-signed with the Developer ID identity: that copy satisfies the installed app's designated
  requirement, so it inherits the folder and Photos grants and raises no TCC prompt (the Apple
  Development build does, and nobody may answer those for Han).
- READ THE STALLS BY WHAT RAN IN THEM. XCUITest serves every query and keystroke through the app's
  accessibility tree on the main thread: ~75 ms per keystroke with no app code in it, and 1-3 s per
  scroll on a 1,158-row folder listing (64% of the samples in accessibility). Temporary
  `UIProbe.count("body.X")` lines in view bodies attach counts to each stall; a stall with no
  bodies is the harness. A `sample` taken during the tour perturbs it enough to time XCUITest out.
- EVERY KEYSTROKE RE-RENDERED THE WINDOW: the split view, the results list, every visible row, the
  sidebar and the whole menu bar, ~60 ms a character. Causes, each fixed: `.searchable`'s binding
  and suggestions read the query from ContentView's body (now the `SearchField` modifier and
  `QuerySuggestions` view); the sidebar's `onChange(of: rawQuery)` (now `RawQueryWatcher`);
  `hasQuery`/`hasActiveSearch`/`currentSearchIsBookmarked` computed from the query and read by the
  window and the menu bar (now stored, written only on change); `@State` typing timers.
- AN @Observable PROPERTY NOTIFIES ON EVERY WRITE, EQUAL OR NOT. Unguarded no-op writes were the
  other half: `recomputeResults` (twice per search), `applyResults` on a same-query refresh,
  selection clears, `queryError = nil`, the 1.5 s index-stats tick (`assign(_:_:)`), and the
  results list's own `@State` resets. `SearchHit`/`ResultGroup`/`DiskUse.Entry` are Equatable for this.
- THE ROW'S CONTEXT MENU IS PART OF THE ROW. macOS builds it eagerly, so what the menu reads
  re-renders the row: it read `rawResults` (which changes below the threshold on nearly every
  keystroke) for a multi-selection check. `ResultsList` takes no arguments and is `.equatable()`,
  so only observation re-renders it: one list pass per result set instead of two or three.
- MEASURED AND NOT DONE: dropping `.id(resultsToken)` halves row renders per new result set (27
  instead of 54 on the real index) but only moves typing stalls 11.3 -> 9.9 s, and the id fixes a
  real scroll-position bug. Empty context menus save ~10% (12.0 -> 10.7 s): not worth lazy menus
  and their VoiceOver risk. Fetching the grouping inputs inside the search task would remove the
  second publish but costs 440-620 ms of cold vector reads before any result shows on the real
  index; grouping stays a refinement after the list is up.
- FOLDER BROWSER (fixed from reading the code, the harness cannot see them): the listing was sorted
  in `body` on every render (now `sorted` in state), the counts were applied one element at a time
  to a `@State` array, row icons hit the icon services daemon per row per render (now a per-path
  cache, keeping Downloads' own icon), each child `URL` stat'ed its path, and the 500 ms ring
  sampler walked every subfolder with nothing indexing (`mayHaveBrowseProgress`).
- NUMBERS, main-thread blocked time over the tour, same clone, back to back. Real index (2.7M
  files, indexing paused): 0.13.8 49.4 s, new 37.1 s; typing 32.2 -> 22.8 s, worst stall
  2352 -> 370 ms; Find Similar 8.1 -> 5.5 s. Scratch corpus while indexing 5.7k new files: 28.4 ->
  19.2 s; typing 12.0 -> 7.0 s, Find Similar 6.6 -> 3.2 s. Both include the harness's own cost.
- FOUND, NOT FIXED: the filename index (`LexicalIndex.rebuildIfStale`) rebuilds from scratch on
  launch whenever the store changed since it was built: ~80 s of one core on the 2.7M index, with
  filename matches absent from search until it finishes. And the first search after launch is
  2.4-3 s on that index (cold), against ~250 ms warm. UPDATED: the first search is fixed
  (0.14-0.22 s) and the rebuild no longer blocks the store queue (index.md, "Launch and the first
  search"); the rebuild's own ~80 s has not been re-measured.
  FIXED 2026-10-03, and it was worse than written: the stamp is the store's persisted mutation
  counter, so EVERY launch after a session that wrote anything rebuilt all names (59.8 s on the
  2.68M-file bench index, the channel off throughout), and nothing refreshed it during a session,
  so a file indexed after launch could not be found by name at all (shipped 0.14.5: the full name
  of a file added mid-session returned other files). Now a DIFF (`refreshIncrementally`): 1,000
  added and 500 removed in 1.39 s on the bench index, identical path set and identical ranked
  top-24 for 300 queries against a fresh build. It runs at launch, after every pass or reconcile,
  and once a minute during a long pass. Needs SQLite `contentless_delete` (3.43+); without it the
  sidecar keeps the old full-rebuild behaviour. The table layout changed, so each existing sidecar
  rebuilds once (~55 s) on the first launch of this build.
- DEV AND TEST LAUNCHES MOVED THE REAL APP'S FSEVENTS CHECKPOINT. `-omni.dbDir` in the argument
  domain changes what is read, not where `UserDefaults.set` writes, so every UI test and scratch run
  wrote `omni.fsEventId` into the user's defaults, and the installed app then resumed past changes
  it had never indexed. Now kept in memory when `-omni.dbDir` is a launch argument
  (`eventCheckpointIsShared`). Negative control: 0.13.8 on a scratch index moved it
  578500264 -> 584312543; the fix left it alone. An earlier checkpoint is the safe direction to
  repair toward: it only replays more events.

## Chaos over every surface (2026-09-23, Scripts/chaos-run.sh, UITests/FullChaosUITests)
- THE TOOL: seeded chaos over the search box, qualifier chips, the sidebar, the folder browser,
  results in both views, the OCR workspace (it opens one-page corpus files through its own panel),
  every toolbar control and the menu bar, with files added and removed under the corpus all run.
  `chaos-run.sh <corpus> <index> <out> [reserve]` records the CHAOS action trail, the stall log, the
  perf log, footprint, crash reports and the app's error/fault os_log (`/usr/bin/log`: in zsh `log`
  is a builtin and the capture silently recorded nothing), and SAMPLES THE APP whenever the trail
  goes quiet for 20 s - a stall is only written when it ends, which a real hang never does.
- READ IT WITH THE HARNESS IN MIND. XCUITest's accessibility queries run on the app's main thread:
  58-68% of the busy time during a folder-listing stall, 2.8 s + 1.2 s per keystroke over 1,156 rows.
  Anything a chaos run times has to be re-measured without it: `OMNI_PERF_SCRIPT` (App/PerfScript.
  swift) runs `browse:/view:/sidebar/search:/clear/wait:` steps in-process and logs each step's
  main-thread CPU. Sidebar toggle over that folder is ~190 ms there, not the 4 s the chaos run saw.
- FIXED, found by chaos:
  - A HANG: the results marquee published row frames through a GeometryReader preference while
    active, and in a lazy stack those values never settled - minutes in LazyStack placement with
    ResultItemFramesKey in every sample, until XCUITest gave up on the app. A click whose mouse-up
    never arrived left the marquee active. Rows now write frames with `onGeometryChange` into a
    reference (`ResultFrames`); the band is `@GestureState`, which cancellation resets.
  - `PhotoLibrary.cleanExportScratch()` listed $TMPDIR on the main thread at every launch: 2.7 s.
  - `refreshDeniedRoots` assigned `deniedRoots` every stats tick; the sidebar reads it.
  - (Superseded 2026-10-03 by `SourceSection`, see "THE PER-PAGE STALL" above.)
    OCR source pane while streaming: one Text of the whole page's highlighted source, re-measured
    and redrawn 24 times a second - main thread 93% busy on a 3,702-token page. One Text per line
    while a page runs (a fence stays whole; HTML tables have no blank lines, so paragraphs were not
    enough), one Text when it finishes: 45%. Same tok/s (362 against 364 with one Text, interleaved).
  - The OCR toolbar was `.toolbar` on the workspace body, which re-runs per streamed update, so every
    update rebuilt the platform toolbar items (~1,000 samples a minute in item layout). It is the
    `OCRToolbar` modifier now, reading only what it shows.
  - HangWatch ran its timer in the default mode only, so an open menu read as a main-thread block
    for as long as it stayed open (112 s once, main thread idle). Common modes now.
- Opening a file that is already open made a second tab over the same pages and queued them again
  (two "Invoice (4).pdf" tabs in a chaos screenshot). It now selects the open tab, as Preview does,
  and a drop naming one file twice opens it once. `-omni.ocrOpen a::b` opens a and b as two drops.
- WATCH THE SCREEN, NOT JUST THE TRAIL. A run logged 44 typed queries, clicks and scrolls over two
  minutes while screenshots showed the same frame: Help > Keyboard Shortcuts was in front, and
  `app.windows.firstMatch` is the FRONT window, so every action went there or found nothing. The
  suite now addresses the main window by title, closes other windows before each action, and fails
  unless typed queries reach the box (43 of 43 on the run after). Two more ways it lost the app:
  opening a result handed it to Preview, and Edit > Start Dictation raised a system dialog; both
  are denied now. A file panel left open by a random toolbar click made every accessibility query
  wait on the panel's remote service (413 s on one click), so only the OCR-open step opens panels.
- NOT BUGS, checked: Escape in the toolbar search field ends the search and gives up focus (native,
  Notes and Mail do it; the app's own escape handler never runs). "prevented access of index N in
  preferredHeights" follows the system's Window > Move & Resize submenu. The Security and Hang Risk
  runtime issues are framework-side or the engine gate's known, boosted inversion. Opening Settings
  is 150-190 ms; the open panel's first show waits ~0.6 s on its out-of-process service.
- A RARE HANG, FIXED BY MEASURING WHAT RESIZES: typing `-type:text porsche` in list view over results
  left the main thread 100% busy for 30 s+ in SwiftUI's lazy prefetch loop
  (`LazyLayoutViewCache.signalPrefetch` -> transaction -> placement -> prefetch), no row bodies in
  the sample. Seen twice, then not in ~690 chaos actions, so no repro to A/B. WWDC26 session 321:
  lazy-stack rows must not change size after they appear. Logging every row whose height changed
  after its first layout (`row-height-changed`, OMNI_PERF_LOG) found two that did: a row becoming a
  stack when grouping lands (the badge made the title line 61 -> 63 pt) and an image row whose tags
  arrived after the search (52 -> 61 pt, the snippet line appearing). The badge no longer adds
  height and the snippet line is always there, blank when empty: 0 rows change over 13 searches in
  list view and 9 in the gallery, against 1 and 2 with the same probe before. Ruled out on the way,
  measured: duplicate ForEach ids and width-dependent heights. The replay is
  `testTypingANegatedQualifierAfterSidebarToggles` (`OMNI_REPRO_ROUNDS` repeats it in one launch).
- OPEN: AppKit's once-per-process "layoutSubtreeIfNeeded on a view which is already being laid
  out" follows a browse step in most runs; lldb on `_NSDetectedLayoutRecursion` did not catch it.
  The rest of a streaming page's cost is window layout AppKit does for the toolbar on each change.

## What the overnight chaos run found (2026-09-30)

- A CRASH: sharing an OCR transcript (File > Share...) trapped in the main-actor isolation check.
  `TranscriptFile.markdown` is built lazily for performance, and the share machinery calls it on a
  background thread. The closure is `@MainActor` now and the export hops to the main actor. It only
  fires when a share service actually asks for the data, so 28 earlier shares had not shown it.
- THE CHAOS SUITE HAD NEVER DRIVEN THREE SURFACES, while its trail said it had:
  - sidebar rows: they matched an outline labelled 'Sidebar' (it has no label) and required
    `isHittable` (a SwiftUI sidebar row is never hittable; its text is). Found by its Recents row
    now, clicked through its text.
  - context menus: `app.menus.firstMatch` is the APPLE MENU, always in the tree. Closed, its items
    were not hittable, so every "choose from the context menu" pressed Escape - and the same lookup
    could have reached Sleep or Restart. The open menu is the one whose items have a frame, and the
    Apple menu's items are denied by name.
  - the four OCR documents it opens by path must exist in the corpus; without them the step only
    held a file panel open, which stalls every accessibility query.
- HEAP RELIEF RAN ON EVERY COVERAGE STAMP. `dropV4TablesLocked` and `buildChunkSplitLocked` put
  `releaseFreedHeap` in a `defer` at their entry, so on a v5 index, where both return at their first
  guard, every stamp walked the malloc zones: ~200 times in 15 minutes of chaos. Relief now runs
  only once a drop or a build has actually happened (8 calls in the next run, all real).
- THE CLIP COUNT LISTED THE FOLDER ON THE MAIN THREAD. `clipboardHasClips` counted files on disk,
  and the Clipboard row's context menu, which macOS builds on every sidebar render, reads it twice:
  7 ms a listing at 3,000 clips. It is a stored count now, recounted off the main thread on launch,
  on a stored clip, after Clear and retention, and whenever the folder's indexed count moves.
- CLEAR DELETED THE FOLDER AFTER RECREATING IT. The delete ran detached and the recreate on the main
  thread, so the delete usually won and capture was left on with no folder. One task, in order.
- SUGGESTIONS CARRIED DUPLICATE IDS. The list is keyed by completion, and a past search can equal a
  completion offered above it (`type:image`); SwiftUI logged "the ID type:image occurs multiple
  times". Deduplicated, first wins.
- A CHAOS RUN WITH 102 STALLS had them in one two-minute burst of back-to-back ~300 ms blocks at
  90-100% CPU, starting at a confirmed Clear. The action trail never went quiet, so the stuck
  sampler never fired and there is no stack. chaos-run.sh now also samples a burst of 8+ CPU-bound
  stalls in 10 s (`hot-<time>.txt`). The first one it caught (34 hot stalls, max 731 ms) was
  XCTest: 42% of the main thread in `XCTElementSnapshotRequest` -> `AXUIElementCopyHierarchy`, 42%
  idle in the event loop, no app frame among the heavy ones. So a hot burst in a chaos run is the
  harness until a sample shows app code; the Clear burst was most likely the same, not proven.
  The sampler names the main thread "Main Thread", not "com.apple.main-thread" - match both.
- DO NOT LAUNCH A SECOND APP WHILE A CHAOS RUN IS GOING. Its window takes focus from the suite and
  its pasteboard poll records the suite's copies; one run failed that way. Check between runs.
- THE RARE LAZY-STACK HANG (results list, `LazyLayoutViewCache.signalPrefetch`), what is now known.
  The loop is SwiftUI's run-loop observer flushing transactions that never drain
  (`GraphHost.flushTransactions` -> `runTransaction`, prefetch enqueuing the next). Seen 5 times in
  ~40 chaos runs; seed 761840 reproduces it in 1 of 5 two-minute replays, always in LIST view. The
  only app frame inside the loop is `ReportResultFrame`'s `onGeometryChange` measuring a row in the
  marquee's coordinate space, which sits OUTSIDE the scroll view, so every row's frame changes on
  every scroll and every placement; about 3.5% of the samples. Suspect, not proven.
  RULED OUT, measured: rows changing height (the probe keys on view and path and logs nothing before
  a hang); accessibility traversal in the hang samples (none).
  DOES NOT REPRODUCE IN PROCESS (2026-10-01, `OMNI_PERF_SCRIPT` with a POPULATED list): 400 result
  sets under 534 posted trackpad scroll gestures with momentum; 200 rounds of new result set,
  sidebar toggle twice, window resize, end/home, select, Settings every tenth round, with scroll
  gestures, both with and without a Swift walker reading the whole accessibility tree every 0.3 s.
  No block over one second in any of them. So whatever starts it is something only XCUITest does.
  The next experiment needs the UI harness: A/B a candidate against seed 761840 with ~15 replays an
  arm to separate 1-in-5 from 0 (~90 min; needs the automation password, drives the real cursor).
  2026-10-02: the duplicate scroll-to-top Task (the leading candidate) was removed for speed, so the
  replays now test it for free: 761840 hanging again means it was not the cause.
- A STRESS SCRIPT ON THE SCRATCH CORPUS MUST USE `score:1%`. Queries with a counter appended score
  under the default 50% threshold there, the results region shows "No results above 50%", and the
  list under test never exists. Two stress runs (150 and 200 rounds) "passed" that way before a
  screenshot showed it. Capture the window once before believing a stress result.
- THE BURST OF STALLS AFTER A CLEAR is not Clear. In process, with 60 clips browsed and then
  cleared: Clear costs 477 ms of main thread, the minute after it 283 ms in total and no block over
  100 ms. The one burst the hot sampler caught in a chaos run was XCTest's snapshot requests.
- A SEARCH COSTS ABOUT ONE SECOND OF MAIN THREAD on the scratch corpus (1694 / 1088 ms for two
  consecutive `score:1%` searches, Release). Sampled: 759 of 2162 samples in window layout, 343 of
  them `NSToolbarView layout` (the toolbar's items change with the result state), and CoreText.
  Taken down 19% on 2026-10-02: see "A RESULT SET COSTS 19% LESS" under the speed review.

## The benchmark (2026-10-08, Settings > Performance > Benchmark this Mac)
- ONE benchmark. It replaced two: the public "Run benchmark" (a downloaded 300-file dataset, one
  indexing pass, files/s and tokens/s) and the hidden "Paper" suite (Option-click, about 1.9 h of
  case budgets though its UI said 25 minutes). Same runner, levers, PaperFS rules and report as the
  suite (`Sources/OmniKit/Paper/`); `omni-verify bench <modelDir> [--scale F]` runs it headless.
- EVERYTHING IS GENERATED from a seed when the button is clicked (`PaperCorpus`, `bench-corpus-2`):
  600 text files, 4,000 tiny files for the crawl, 48 PNGs, 12 WAV clips (integer synthesis), 6 MP4
  clips, and a one-million-row store of seeded vectors (`BenchStore`: 250k files, one in eight an
  image, one in 5,000 a 3,000-row file, a tenth of passages from a shared pool, eight top folders).
  The manifest hash covers every byte but the MP4s, whose container bytes the system encoder owns.
  Nothing reads the user's index or files any more: the live family (p13-p18) is gone.
- THE TABLE (`BenchTable`): Indexing, Queries, Search under load (a search every 50 ms on a copy of
  the store while one bulk change runs: rewrite 5,000 files, delete 50,000, remove a top folder,
  remove every image, reclaim), Mechanisms. The result sheet shows it; Copy and Save take the full
  report; the upload (consent, as before) is `BenchUpload`: the table plus the old top-level
  throughput fields, under the collector's 8 KB limit.
- CASES DROPPED: p05 edit reuse (never emitted its result: arms renamed under it), p19 cap sweep
  (the representation no longer depends on the cap), p12 media tagging (toggled a setting nothing
  reads), p04/p06 (retired already), the live family; p03's chunk-count check (content-defined
  chunking ended the fixed grid it predicted). Recall now grids one and three bits (three is the
  shipped affine width); selection keeps only the shipped form and the primitive.
- Smoke at `--scale 0.05`: 16 cases ok in 2 min 20 s. Full run on the M3 Ultra: 677-690 s. The
  budgets sum to 4,795 s, the ceiling on any Mac; Settings quotes "10 to 80 minutes" until the
  smaller Macs have measured numbers.
- bench-v5 (0.15.10) after the first in-app run (M3 Ultra, 0.15.9) showed two rows were wrong:
  - PER-FILE REUSE read 1.9%: its off arm turned off only the per-file cache and left the
    cross-file one on, so both arms reused every unchanged chunk (7.41 vs 7.27 ms). Both off:
    38.8 ms against 7.95 ms, 79.5% saved.
  - SHAPING read +55% on one run and -104% on the next: 120 searches an arm in sequence, so p99
    was the second-largest sample and whichever arm drew two stray 30-70 ms searches lost. Now 4
    rounds, arm order rotating, 400 an arm, and the no-ceiling arm (no row, no paper number) is
    gone: 64%, 35%, 27% over three runs, the case 73 -> 162 s.
  - Every latency row now carries its worst sample (the p50's largest run) instead of "max -".
  The collector aggregates v4 runs for every row but those two.
- `--only store_build,search_under_writes` runs a subset (2 min); the report marks it SUBSET RUN.
  With OMNI_SEARCH_TIMING=1 every slice, checkpoint and slow search logs its phases.
- WHAT IT FOUND ON ITS FIRST FULL RUN: a 1.3-1.8 s stall in the bulk delete (tombstones over budget
  compacted inside the delete), the WAL valve on the store queue, and an ungated folder tail
  (index.md, "Heavy CRUD"). The store is exact below a million contents on the M3 Ultra, so
  "While reclaiming space" reports not applicable there: no coverage, nothing to relocate.
