import SwiftUI
import AppKit

/// Invisible helper that installs a local key monitor so the space bar triggers Quick Look
/// for the selected result - in both list and gallery views, regardless of which subview
/// holds focus. A focus-based `.onKeyPress` is unreliable here because the List swallows
/// the space key before an ancestor handler sees it. Space is left untouched while editing
/// text (e.g. the search field), so typing a space there still works.
struct QuickLookKeyMonitor: NSViewRepresentable {
    /// Invoked on the main thread when space is pressed outside a text field.
    let onSpace: () -> Void
    /// Invoked for an arrow key while Quick Look is open (delta -1 = up/left, +1 = down/right).
    /// Returns true if it navigated the preview - then the key is consumed. The panel is the key
    /// window while open and a single-item preview ignores arrows, so the app must drive them.
    let onPreviewArrow: (Int) -> Bool

    func makeNSView(context: Context) -> NSView {
        context.coordinator.onSpace = onSpace
        context.coordinator.onPreviewArrow = onPreviewArrow
        context.coordinator.install()
        return NSView(frame: .zero)
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.onSpace = onSpace
        context.coordinator.onPreviewArrow = onPreviewArrow
    }
    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.uninstall()
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var onSpace: (() -> Void)?
        var onPreviewArrow: ((Int) -> Bool)?
        private var monitor: Any?

        func install() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self else { return event }
                // Local key monitors fire on the main thread, so the main-actor responder read is
                // safe to assume isolated; only the Bool result crosses back to this closure.
                let editingText = MainActor.assumeIsolated { NSApp.keyWindow?.firstResponder is NSText }
                if editingText { return event }
                switch event.keyCode {
                case 49:                          // space: toggle Quick Look
                    self.onSpace?()
                    return nil
                case 125, 124:                    // down / right: next preview item
                    return (self.onPreviewArrow?(1) ?? false) ? nil : event
                case 126, 123:                    // up / left: previous preview item
                    return (self.onPreviewArrow?(-1) ?? false) ? nil : event
                default:
                    return event
                }
            }
        }
        func uninstall() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }
    }
}
