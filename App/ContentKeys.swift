import AppKit
import OmniKit
import SwiftUI

/// What a content view (search results, folder / Photos / Recents browser) offers the keyboard: its
/// items in display order, which one is active, and how many columns they are laid out in.
@MainActor
struct KeyNav {
    var count: Int
    var active: Int?
    /// 1 in a list. In a grid, the columns as LAID OUT (see `GridColumns`).
    var columns: Int
    /// Display name of item i, for type-select.
    var name: (Int) -> String
    /// Make item i active; `extend` is Shift held (a range, where the view supports one).
    var select: (_ index: Int, _ extend: Bool) -> Void
    /// Return / Cmd-Down on the active item: a file opens, a folder is entered.
    var open: () -> Void
    /// Right / Left in a LIST (true = right). Returns false to leave the key alone.
    var disclose: ((Bool) -> Bool)? = nil
    var quickLook: () -> Void
}

/// THE CONTENT PANE'S KEYBOARD, the way Finder's content view has it, for every view that lists
/// files, in both layouts. One monitor and one set of rules, because four views each wiring their
/// own had drifted: the browsers' grids took no arrows at all, their lists only while the List held
/// focus, and the results only while SwiftUI had focused the scroll view, which a click on a row
/// does not do - so arrows went to the sidebar, or to the search field, or nowhere.
///
/// A LOCAL MONITOR, not focus: it takes the keys whenever the main window is key and no text is
/// being edited, including while the sidebar holds focus. Finder's sidebar does not move on
/// arrows either - they belong to the files. Cmd-Up is left to the Go menu's Enclosing Folder.
///
///   arrows             move (grid: by visual row and column, no wrapping, Finder's rules)
///   Shift-arrows       extend, where the view keeps a multi-selection (search results)
///   Opt-Up/Down, Home/End   first / last
///   Return, Cmd-Down   open the active item (a folder is entered)
///   Space              Quick Look
///   letters            select the first item whose name starts with what was typed
struct ContentKeyMonitor: NSViewRepresentable {
    let nav: @MainActor () -> KeyNav?
    let isPreviewOpen: @MainActor () -> Bool

    func makeNSView(context: Context) -> NSView {
        context.coordinator.nav = nav
        context.coordinator.isPreviewOpen = isPreviewOpen
        context.coordinator.install()
        return NSView(frame: .zero)
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.nav = nav
        context.coordinator.isPreviewOpen = isPreviewOpen
    }
    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) { coordinator.uninstall() }
    func makeCoordinator() -> Coordinator { Coordinator() }

    /// A click on an item is a move INTO the content, and the keyboard goes with it, the way
    /// Finder's does. The search field gives it up, or the arrows that follow edit the query; and
    /// so does the sidebar, whose Delete removes a whole folder from Omni and stayed armed while the
    /// user was clicking files beside it.
    @MainActor static func takeKeyboard() {
        guard let w = NSApp.keyWindow, w.firstResponder is NSText || w.firstResponder is NSTableView
        else { return }
        w.makeFirstResponder(nil)
    }

    @MainActor final class Coordinator {
        var nav: (@MainActor () -> KeyNav?)?
        var isPreviewOpen: (@MainActor () -> Bool)?
        private var monitor: Any?
        private var typed = ""
        private var typedAt = Date.distantPast

        /// The installed one, for PerfScript's `key:` step. One content view is mounted at a time.
        static weak var current: Coordinator?

        func install() {
            Self.current = self
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self else { return event }
                // The parts that are read, copied out: NSEvent is not Sendable.
                let key = Key(code: event.keyCode, flags: event.modifierFlags, chars: event.characters)
                return MainActor.assumeIsolated { self.handle(key) } ? nil : event
            }
        }
        func uninstall() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }

        struct Key: Sendable { let code: UInt16; let flags: NSEvent.ModifierFlags; let chars: String? }

        /// PerfScript's `key:` step: the same handler, minus the key-window test, which a run
        /// launched in the background cannot pass without taking the user's focus.
        func dispatchForScript(code: UInt16, flags: NSEvent.ModifierFlags, chars: String?) -> Bool {
            handle(Key(code: code, flags: flags, chars: chars), scripted: true)
        }

        private func handle(_ event: Key, scripted: Bool = false) -> Bool {
            let previewOpen = isPreviewOpen?() ?? false
            let inMain = scripted || (NSApp.keyWindow?.identifier?.rawValue.hasPrefix("main") ?? false)
            // The Quick Look panel is the key window while open; the app's own state says so.
            guard inMain || previewOpen, let nav = nav?() else { return false }
            if inMain, !scripted, NSApp.keyWindow?.firstResponder is NSText { return false }   // editing text
            let mods = event.flags.intersection(.deviceIndependentFlagsMask)
                .subtracting([.numericPad, .function, .capsLock])
            let shift = mods == .shift
            let plain = mods.isEmpty
            switch event.code {
            case 49 where plain:                              // space
                nav.quickLook(); return true
            case 36 where plain, 76 where plain:              // return, enter
                guard nav.active != nil else { return false }
                nav.open(); return true
            case 125 where mods == .command:                  // Cmd-Down
                guard nav.active != nil else { return false }
                nav.open(); return true
            case 126 where mods == .option, 115 where plain:  // Opt-Up, Home
                return jump(nav, to: 0)
            case 125 where mods == .option, 119 where plain:  // Opt-Down, End
                return jump(nav, to: nav.count - 1)
            case 125 where plain || shift: return move(nav, by: nav.columns, vertical: true, extend: shift)
            case 126 where plain || shift: return move(nav, by: -nav.columns, vertical: true, extend: shift)
            case 124 where plain || shift, 123 where plain || shift:
                let right = event.code == 124
                if nav.columns <= 1 {
                    guard plain, let disclose = nav.disclose else { return false }
                    return disclose(right)
                }
                return move(nav, by: right ? 1 : -1, vertical: false, extend: shift)
            default:
                guard plain || shift, !previewOpen, let chars = event.chars, !chars.isEmpty,
                      chars.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }),
                      chars != " " else { return false }
                return typeSelect(nav, chars)
            }
        }

        private func jump(_ nav: KeyNav, to i: Int) -> Bool {
            guard nav.count > 0 else { return false }
            nav.select(max(0, min(nav.count - 1, i)), false)
            return true
        }

        /// Finder's rules. Nothing active: forward takes the first item, backward the last. A list
        /// clamps at its ends. A grid moves sideways only within the visual row, and vertically only
        /// while a row exists that way - Down into a shorter last row lands on its last item.
        private func move(_ nav: KeyNav, by delta: Int, vertical: Bool, extend: Bool) -> Bool {
            guard nav.count > 0 else { return false }
            guard let cur = nav.active else {
                nav.select(delta > 0 ? 0 : nav.count - 1, false)
                return true
            }
            let cols = max(1, nav.columns)
            if omniPerfEnabled { omniPerfLog("keys move from=\(cur) by=\(delta) columns=\(cols) count=\(nav.count)") }
            let target = cur + delta
            var next: Int?
            if cols == 1 {
                next = max(0, min(nav.count - 1, target))
            } else if !vertical {
                if target >= 0, target < nav.count, target / cols == cur / cols { next = target }
            } else if target < 0 {
                next = nil
            } else if target >= nav.count {
                if cur / cols < (nav.count - 1) / cols { next = nav.count - 1 }
            } else {
                next = target
            }
            if let next, next != cur { nav.select(next, extend) }
            return true   // consumed even at an edge, so the key never falls through to the sidebar
        }

        /// Finder's type-select: letters typed within a second of each other build one prefix.
        private func typeSelect(_ nav: KeyNav, _ chars: String) -> Bool {
            let now = Date()
            if now.timeIntervalSince(typedAt) > 1 { typed = "" }
            typed += chars.lowercased()
            typedAt = now
            let prefix = typed
            if omniPerfEnabled { omniPerfLog("keys type-select \(prefix)") }
            guard let i = (0 ..< nav.count).first(where: { nav.name($0).lowercased().hasPrefix(prefix) })
            else { return true }
            nav.select(i, false)
            return true
        }
    }
}

/// The column count a `LazyVGrid(.adaptive(minimum:))` actually lays out, from the width it is
/// given. SwiftUI fits as many minimum-width items with their spacing as the width allows, so this
/// is exact where the old estimate from the SCROLL VIEW's width was not: a visible scroller takes
/// its width from the grid and not from the scroll view, and near a boundary the estimate was one
/// column off - Up and Down then landed in the wrong column.
enum GridColumns {
    static func count(width: CGFloat, minimum: CGFloat, spacing: CGFloat) -> Int {
        guard width > 0 else { return 1 }
        return max(1, Int((width + spacing) / (minimum + spacing)))
    }
}
