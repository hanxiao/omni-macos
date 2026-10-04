# Clipboard history (2026-09-29)

Every clipboard change becomes a real file in an app-managed folder, and that folder is an indexed
root with its own sidebar row. Search, filters, the folder browser, thumbnails, Quick Look, Find
Similar and deletion are the existing pipeline, unchanged. No schema change, no migration.

A virtual `clipboard://` ContentSource was considered: Photos shows its cost (own browser, own
thumbnails, own reconcile rules), and the pasteboard holds only the latest item, so history has to
be persisted somewhere anyway. Files are that storage.

## Decisions

- Opt-in. Off until enabled from the sidebar or Settings > Files.
- Text and images. Plain and rich text stored as `.txt`, images as `.png`, URLs as text. Finder
  file copies are skipped: they are already files.
- Always skipped: items marked `org.nspasteboard.ConcealedType` or `org.nspasteboard.TransientType`
  (what password managers set), and copies made inside Omni.
- Retention 30 days by default; 7 / 30 / 90 days / forever in Settings.
- App only. Clipboard files are excluded from served search (HTTP and MCP) and from
  `list_sources`.

## Parts

1. `App/ClipboardMonitor.swift`: polls `NSPasteboard.general.changeCount` (about 2 Hz; there is no
   change notification), picks the first item's best type, applies the skips.
2. Store: writes `YYYY-MM-DD HH.MM.SS <first words>.txt|.png` into `Application Support/Omni/Clipboard`
   (beside the index when `-omni.dbDir` is a launch argument). The same content again bumps the
   existing file's date instead of writing a new file. Retention deletes old files; the watcher
   removes their rows.
3. Index: the clipboard folder joins the root list when enabled, outside `addedFolders` so it is
   not drawn as a user folder. It is exempt from the ignore rules, whose default `Library/` would
   otherwise exclude it.
4. UX: a "Clipboard" row under the Photos rows in the Index section, ALWAYS present. With capture
   off (never opted in, or turned off) clicking it shows the standard empty-state screen: the
   monochrome mole through `StatusGlyph`, a title, and one button that turns capture on. The search
   is scoped to the clipboard folder there too (`in:Clipboard`), so a query typed on that screen
   finds nothing and the screen stays; it used to clear the scope and search every folder. With
   capture on it is a regular folder in every respect: the folder browser (list and gallery,
   columns, sort, selection, context menus), search scoped to it, filters and chips, back/forward.
   Newest first by default. Context menu: Pause capture, Clear clipboard history..., Show in
   Finder. Settings > Files, ahead of the folders: "Index clipboard content" and "Keep content for";
   Clear is on the sidebar row's context menu only. UI text follows the UI rules in
   CLAUDE.md: a title and one factual clause at most, no explanation of internals.
5. Serving: served search drops hits under the clipboard folder.

## Verification

- Unit: type mapping, concealed/transient skip, duplicate bump, naming, retention.
- End to end, isolated run (`-omni.dbDir`, `-omni.addedFolders`, `-omni.ephemeralUIState`,
  `-omni.serving.port 51299`): `pbcopy` text and copy an image, check each file appears and is
  found by in-app search; a concealed-marked copy leaves no file; a served search does not return
  clipboard hits.

## Built, and what building it found

- A QUERY MADE FROM THE CLIPBOARD IS NEVER ANSWERED BY ITS OWN CLIP. Pasting text into the box or
  an image into the window searches with exactly what the newest clip holds, so that clip ranked
  first at the query's own score, above the document the text came from. `applyResults` drops the
  clip of the current clipboard content when the query is a paste (`FileQuery.fromPasteboard`) or
  its text equals that clip's text. Compared against what the monitor stored, never by reading the
  pasteboard again: macOS 15.4+ can gate programmatic reads behind a prompt. At launch the monitor
  also looks the clipboard's content up by digest (`existingFile(for:)`), so text copied in an
  earlier session and pasted in this one does not find its own clip either.
- Measured: with the clipboard holding a sentence whose clip is indexed, searching that exact
  sentence returns the source document and not the clip; once the clipboard holds something else,
  the same search returns the clip first, and a paraphrase finds it in both cases.
- NO CLIPBOARD WATCHER. The folder joins the crawl roots, and the existing FSEvents watcher,
  reconcile and indexer do everything; the only clipboard-specific loop is the pasteboard poll,
  because macOS posts no change notification.
- OMNI'S OWN COPIES ARE MARKED, NOT GUESSED. The first version skipped any copy made while Omni was
  frontmost, which dropped everything a user copies from another window while Omni has focus, and
  everything a UI test copies. Every copy Omni makes goes through `OmniPasteboard.copy`, which adds
  `io.hanxiao.omni.own`; the serving token also carries the nspasteboard.org concealed type.
- THE CLIPBOARD FOLDER LIVES IN OMNI'S OWN DATA and the crawl skips own data, so it is listed in
  `ownDataExceptions`. It also lives under `~/Library`, and the watcher's ancestor check tested the
  folders ABOVE a root against the default `Library/` rule, so every clip event read as excluded and
  was deleted. That was a real bug for any root under `~/Library` - iCloud Drive included - fixed at
  the source: `isIgnoredIncludingAncestors(root:)` and `excludesIndexedFile(roots:)` stop at the
  root, as the crawl always did. `IgnoreRootBoundTests` pins it, and asserts the old answer first.
- `roots` STAYS THE USER'S FOLDERS; `crawlRoots` adds the clipboard. The sidebar, Settings, the Go
  menu, `in:` suggestions and the served root check (`omni.roots`) all read `roots`, so none of them
  lists the clipboard; the indexer, the watcher and `rootKey` read `crawlRoots`.
- SERVED SEARCH excludes the folder through `SearchFilter.excludeFolders`, resolved into the same
  deny set `-tag:` uses. `ExcludeFoldersTests` has a negative control (2 failures with it off).
- Measured end to end on an isolated run: a text clip, a concealed copy (no file), an image clip,
  the first text again (renamed, one file); all indexed; in-app search returns the clip beside the
  source document; served search returns only the document.
