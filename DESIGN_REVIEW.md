# Omni macOS Design Review

## Executive Summary

Omni already reads as a credible native macOS utility: it uses NavigationSplitView, a real `.searchable` toolbar field, Quick Look, draggable results, and a tabbed Settings window, and the Spotlight-style results list with score percentages feels at home on the platform. The gaps are not in the bones but in the finish: selection chrome, type sizing, materials, and keyboard reachability diverge from how Finder and Photos actually behave, so the app currently feels like a very good approximation rather than something Apple shipped. The three highest-impact fixes are: (1) replace the thick 2.5px accent ring on gallery tiles with a proper translucent selection fill behind the whole cell, the single most visible non-native tell; (2) make Open, Quick Look, Reveal, and view/sort switching reachable from menus and the keyboard with visible shortcut hints, since the primary action currently has no menu command or shortcut; and (3) give the gallery the same arrow-key and Return/Space selection semantics the list gets for free, so switching to gallery does not silently kill keyboard navigation. Secondary but worthwhile: hide (not just disable) toolbars and search during onboarding and empty states, and stop using `ultraThinMaterial` inside the opaque content area. None of these are architectural; they are targeted refinements to existing views.

## High Severity

### 1. Gallery selection is a thick accent ring, not a native selection fill
**Where:** `ResultsList.swift:ResultGridItem`

Selected tiles use `RoundedRectangle(...).strokeBorder(Color.accentColor, lineWidth: selected ? 2.5 : 0)` clipped to the thumbnail edge. A 2.5px hard ring hugging just the image reads like a web focus outline, and it leaves the filename and caption looking unselected. Finder and Photos indicate selection with a translucent rounded accent fill behind the entire cell (thumbnail plus label) and render the filename in a filled accent pill.

**Fix:** Drop the stroke. Wrap the whole `VStack` (thumbnail + name + media info) in a selection background and put the label in an accent pill when selected:
```swift
VStack { /* thumbnail, name, info */ }
    .padding(6)
    .background(
        selected ? Color.accentColor.opacity(0.18) : .clear,
        in: RoundedRectangle(cornerRadius: Design.corner + 2, style: .continuous)
    )
// filename:
Text(name)
    .foregroundStyle(selected ? .white : .primary)
    .padding(.horizontal, 6).padding(.vertical, 1)
    .background(selected ? Color.accentColor : .clear, in: Capsule())
```
Keep the thumbnail's own 0.5px separator for edge definition.

### 2. Primary action (Open) has no menu command and no keyboard shortcut
**Where:** `ResultsList.swift:menu / open(_:)`, `OmniApp.swift commands`

Opening a result, the single most important action, is reachable only by double-click, Return (via `onKeyPress`), or a context-menu item with no shortcut hint. There is no File-menu command and no Cmd-O / Cmd-Down. Finder binds Open to Cmd-O and Cmd-Down and shows the hint inline. Quick Look (Space) and Reveal have the same problem: they work but are undiscoverable because the context-menu buttons carry no `.keyboardShortcut`, so no glyph renders.

**Fix:** Add a File `CommandGroup` driven off `model.selection`, and mirror the same shortcuts on the context-menu buttons so the hints appear in the menu:
```swift
CommandGroup(after: .newItem) {
    Button("Open") { model.openSelected() }
        .keyboardShortcut(.downArrow, modifiers: .command)
    Button("Quick Look") { model.quickLookSelected() }
        .keyboardShortcut(.space, modifiers: [])
    Button("Reveal in Finder") { model.revealSelected() }
        .keyboardShortcut("r", modifiers: [.command, .shift])
}
```
Optionally label Quick Look as `Quick Look "<filename>"` the way Finder does.

### 3. Gallery view loses arrow-key navigation and standard selection
**Where:** `ResultsList.swift:gridView (ResultGridItem + onTapGesture)`

The list uses `List(selection:)`, so Up/Down arrows, type-select, and Return work. The gallery binds selection through a bare `.onTapGesture { model.selection = hit.path }` per tile, with a manual double-click gesture and a hand-drawn selection ring. The moment the user switches to gallery, arrow navigation, type-to-select, range selection, and Return-to-open silently stop. Finder's icon view is fully keyboard-navigable in both layouts.

**Fix:** Make the grid focusable and implement `.onMoveCommand` to move `model.selection` by row and column, reading from the same selection source the existing `onKeyPress(.return)`/Space handlers already use, so navigation and open/preview behave identically in both modes:
```swift
gridView
    .focusable()
    .focusEffectDisabled()
    .onMoveCommand { direction in model.moveSelection(direction, columns: columnCount) }
```
Compute `columnCount` from the current width and the adaptive `GridItem` minimum so left/right and up/down map correctly.

## Medium Severity

### 4. Toolbars and search are shown (disabled or live) during onboarding and empty states
**Where:** `ContentView.swift:toolbar` and `.searchable` on `NavigationSplitView`

The filter menu, sort menu, and list/gallery segmented control are attached unconditionally and only `.disabled()` when there is no content, so they render as greyed chrome around the "Add a folder to search" onboarding screen and the loading/noModel/failed states. Separately, `.searchable` is applied to the whole split view, so a live "Search your files by meaning" field appears even when there are no folders and nothing to search, inviting an action that cannot return anything. The progressive-disclosure rule is explicit: secondary UI should be hidden, not dimmed, until content exists, and search should expand from a collapsed state once there is a corpus.

**Fix:** Gate the toolbar with a `@ToolbarContentBuilder` that returns `EmptyView` unless `model.phase == .ready && !model.rawResults.isEmpty`, and apply `.searchable` conditionally on `model.phase == .ready && model.indexedFiles > 0`. Keep onboarding to its single Add Folder CTA.

### 5. Settings window is one fixed 480x380 frame across five very different tabs
**Where:** `SettingsView.swift:SettingsView (.frame(width: 480, height: 380))`

All five tabs share one hard-coded height. Storage can additionally show an "Index is out of date" banner plus a Model download section, while Content has four toggles. In the screenshot the first form row sits tight under the tab strip and the grouped Form looks cramped at the top, and shorter tabs leave dead space. Apple's preference windows and System Settings size the window to each selected pane.

**Fix:** Drop the fixed height and let each tab drive size, animating between tab-specific heights:
```swift
TabView(selection: $tab) { /* tabs */ }
    .frame(width: 480)
    .fixedSize(horizontal: false, vertical: true)
```
or give each tab its own `.frame(minHeight:)`. At minimum raise the height so the first section header clears the tab bar without clipping.

### 6. Passages panel uses `ultraThinMaterial` inside the opaque content area
**Where:** `ResultsList.swift:PassagesView`

The expanded matching-passages box is `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))`. Blur and vibrancy belong on sidebars, toolbars, and floating panels, not in the main content area, which should stay solid for readability. This is inline content inside the scrolling list, so the excerpt text sits on a blurry, lower-contrast surface that competes with the list background.

**Fix:** Use a flat elevated fill that reads as a nested card:
```swift
.background(Color(.controlBackgroundColor),
            in: RoundedRectangle(cornerRadius: 6, style: .continuous))
// or .background(.quaternary, in: ...)
```

### 7. View-mode and Sort have no View menu or Cmd-1 / Cmd-2 bindings
**Where:** `ContentView.swift:toolbar (viewMode Picker, sortOrder Menu)`

The segmented control and sort menu live only in the toolbar. There is no View menu and no keyboard shortcuts, so a power user cannot switch list/gallery or change sort from the keyboard. Finder offers as Icons (Cmd-1), as List (Cmd-2), and a Sort By submenu, all with inline hints.

**Fix:** Add a `CommandMenu("View")` with toggles bound to `.keyboardShortcut("1")` (gallery) and `.keyboardShortcut("2")` (list) to match Finder, plus a Sort By submenu. Keep the toolbar control as the visual affordance.

### 8. Quick Look (Space) is undiscoverable
**Where:** `ResultsList.swift:QuickLookKeyMonitor / menu(_:)`

Space-to-preview is implemented via a low-level `NSEvent` monitor and works, but it is invisible: the context-menu "Quick Look" button has no `.keyboardShortcut`, so no Space glyph renders, and there is no File-menu equivalent. Users who do not already know the Finder convention will never find it.

**Fix:** Add `.keyboardShortcut(.space, modifiers: [])` to the Quick Look context-menu button and a File > Quick Look command (folds into the File `CommandGroup` from item 2).

### 9. App is draggable out but accepts no drop-in
**Where:** `ResultsList.swift (.draggable)`, `Sidebar.swift:pickFolder`, `ContentView.swift:pickFolder`

Content-out works (rows and tiles are `.draggable` URLs). Content-in is absent: the only way to add a search root is the NSOpenPanel behind the "+"/Add Folder button. There is no `.dropDestination` on the sidebar or the empty state, and no dragover highlight. Dropping a folder from Finder is the most natural way to add a search root on macOS.

**Fix:** Add a drop target on the sidebar Folders section and the empty state, with an `isTargeted` affordance:
```swift
.dropDestination(for: URL.self) { urls, _ in
    let dirs = urls.filter { $0.hasDirectoryPath }
    dirs.forEach(model.addRoot)
    return !dirs.isEmpty
} isTargeted: { hovering = $0 }
.background(hovering ? Color.accentColor.opacity(0.12) : .clear)
```

## Low Severity

### 10. Body and snippet text use `.callout` (12px) instead of `.body` (13px)
**Where:** `ResultsList.swift:ResultRow / PassagesView`, `ContentView.belowThresholdFooter`

The result snippet, passage excerpts, and the "Show N more" footer all use `.font(.callout)` (12px). The text the user actually reads to judge relevance is body-level and Apple sizes that at 13px, so the results feel slightly cramped versus a native list. **Fix:** use `.font(.body)` for the snippet and excerpt text; keep `.callout`/`.caption` for the de-emphasized metadata line.

### 11. Gallery score badge is a fixed non-adaptive black scrim
**Where:** `ResultsList.swift:ResultGridItem score overlay`

The relevance badge is `.background(.black.opacity(0.5), in: Capsule())` with hardcoded white text, identical in light and dark mode. Over bright thumbnails it is too weak, over dark ones too heavy. **Fix:** a material capsule, the one legitimate in-content use of material (a chip over imagery, as Photos does): `.background(.ultraThinMaterial, in: Capsule())` with `.foregroundStyle(.primary)`.

### 12. Thumbnail edge uses `separatorColor` as a uniform 4-sided box
**Where:** `Thumbnail.swift:Thumbnail`

Every thumbnail gets `.strokeBorder(Color(.separatorColor), lineWidth: 0.5)`. `separatorColor` is tuned for list dividers and is relatively visible, so ringing every 40px result and 128px tile draws a grid of hairlines across the content. Finder uses a soft bottom-weighted shadow on thumbnails, not a 4-sided box. **Fix:** use `Color.primary.opacity(0.08)` for the edge, or prefer a soft drop shadow (`0 0 0 0.5px` plus `0 1px 2px`) for image thumbnails and reserve the hairline for non-image file icons.

### 13. Empty-state title bumps Title 2 to semibold instead of using a native style
**Where:** `ContentView.swift:CenteredStatus`

The headline is `.font(.title2).fontWeight(.semibold)`. Title 2 (17px) is a Regular-weight style; forcing semibold onto it diverges from the system weights, and the headline is small for a full-pane empty state. **Fix:** use `.font(.title)` (Title 1, 22px) and let the style carry the weight so it tracks the system and Dynamic Type.

### 14. Selected result has no persistent preview pane
**Where:** `ResultsList.swift (Quick Look / selection)`

Selecting a result either does nothing inline or opens Quick Look as a separate floating window; only text passages expand inline. Inspecting an image or video loses list context the way full-page navigation would. The preferred pattern is a slide-out detail panel from the right that keeps the grid/list visible. **Fix:** consider a third `NavigationSplitView` column or an overlay panel previewing the selected hit (thumbnail/Quick Look content, path, score, reveal/open actions) instead of relying solely on the modal Quick Look window.

### 15. Gallery tiles can shrink to 140pt
**Where:** `ResultsList.swift:gridView (GridItem .adaptive minimum 140)`

The adaptive columns floor at 140pt, which packs many small tiles in a wide window and trends to the dense end of the gallery guideline (2/3/4-5 columns by window size, large breathable media). **Fix:** raise the adaptive minimum to 160-180pt, keeping the existing 16pt gap and padding.

### 16. Cmd-R for Index collides with Finder's Reveal/Reload meaning
**Where:** `OmniApp.swift commands (Index button keyboardShortcut("r"))`

Index/Resume is bound to Cmd-R, but in Finder Cmd-R means Show Original and across macOS Cmd-R generally means Reload. In a file browser this is mildly surprising. **Fix:** leave Index in the menu without a global shortcut or move it to Cmd-Shift-I, and reserve Cmd-R / Cmd-Shift-R for Reveal in Finder (item 2).

### 17. No shortcut cheat sheet or discoverability surface
**Where:** App-wide (`OmniApp.swift commands`, `SettingsView.swift`)

Several interactions are keyboard-only (Space = Quick Look, Return = Open), but there is no Help-menu cheat sheet and no Shortcuts tab in Settings. As a third-party app, users will not assume Finder conventions carry over. **Fix:** add a Keyboard Shortcuts section in Settings or a Help-menu item (optionally Cmd-/) listing focus search, open, Quick Look, reveal, and index.

### 18. Search field is not auto-focused on launch
**Where:** `ContentView.swift (.searchable, no FocusState)`

For a Spotlight-like utility the search field is the primary input, yet focus is not placed there on launch; the user must click the field or invoke Find first. Spotlight and Raycast focus the query field immediately. **Fix:** use `.searchable(text:, isPresented:)` or `@FocusState` with `.searchFocused`/`.defaultFocus` to put the caret in the field when the window becomes key and `model.phase == .ready`.
