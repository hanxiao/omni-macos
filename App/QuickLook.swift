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

    func makeNSView(context: Context) -> NSView {
        context.coordinator.onSpace = onSpace
        context.coordinator.install()
        return NSView(frame: .zero)
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.onSpace = onSpace
    }
    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.uninstall()
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var onSpace: (() -> Void)?
        private var monitor: Any?

        func install() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self, event.keyCode == 49 else { return event }   // 49 = space
                // Don't steal the space bar from text fields (the search field's editor).
                if let fr = event.window?.firstResponder, fr is NSText { return event }
                self.onSpace?()
                return nil   // consume
            }
        }
        func uninstall() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }
    }
}
