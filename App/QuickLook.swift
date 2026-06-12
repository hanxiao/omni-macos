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
    /// Invoked for an arrow key while Quick Look is open (`vertical` = up/down, `forward` =
    /// down/right). Returns true if it navigated the preview - then the key is consumed. The
    /// panel is the key window while open and a single-item preview ignores arrows, so the app
    /// must drive them; the axis lets the gallery move by visual row, not linearly.
    let onPreviewArrow: (_ vertical: Bool, _ forward: Bool) -> Bool
    /// Whether the app considers a Quick Look preview open (model.previewURL != nil). The scope
    /// check below cannot rely on the key window's class: SwiftUI's preview panel is the key
    /// window while open and is not reliably a QLPreviewPanel subclass.
    let isPreviewOpen: () -> Bool

    func makeNSView(context: Context) -> NSView {
        let v = NSView(frame: .zero)
        context.coordinator.onSpace = onSpace
        context.coordinator.onPreviewArrow = onPreviewArrow
        context.coordinator.isPreviewOpen = isPreviewOpen
        context.coordinator.install()
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.onSpace = onSpace
        context.coordinator.onPreviewArrow = onPreviewArrow
        context.coordinator.isPreviewOpen = isPreviewOpen
    }
    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.uninstall()
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var onSpace: (() -> Void)?
        var onPreviewArrow: ((_ vertical: Bool, _ forward: Bool) -> Bool)?
        var isPreviewOpen: (() -> Bool)?
        private var monitor: Any?

        func install() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self else { return event }
                // Local key monitors fire on the main thread, so the main-actor reads are safe to
                // assume isolated; only the Bool result crosses back to this closure.
                // Scope to the main window (and the Quick Look panel, which becomes key while
                // open and whose arrows/space this monitor drives): a global monitor swallowed
                // Space in Settings and any other window, breaking keyboard control there.
                let inMainWindow = MainActor.assumeIsolated {
                    NSApp.keyWindow?.identifier?.rawValue.hasPrefix("main") ?? false
                }
                // While a preview is open the PANEL is the key window (and its class is an
                // implementation detail) - the app's own preview state is the reliable signal.
                guard inMainWindow || self.isPreviewOpen?() == true else { return event }
                let editingText = MainActor.assumeIsolated { NSApp.keyWindow?.firstResponder is NSText }
                if editingText { return event }
                switch event.keyCode {
                case 49:                          // space: toggle Quick Look
                    self.onSpace?()
                    return nil
                case 125:                         // down
                    return (self.onPreviewArrow?(true, true) ?? false) ? nil : event
                case 124:                         // right
                    return (self.onPreviewArrow?(false, true) ?? false) ? nil : event
                case 126:                         // up
                    return (self.onPreviewArrow?(true, false) ?? false) ? nil : event
                case 123:                         // left
                    return (self.onPreviewArrow?(false, false) ?? false) ? nil : event
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
