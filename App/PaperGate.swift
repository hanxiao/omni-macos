import Foundation
import SwiftUI
import AppKit

/// Whether the hidden "Paper" control renders at all.
///
/// Hidden means absent, not disabled: a disabled button still tells every user the feature exists
/// and invites a support question about a 25-minute benchmark they must never run. Three ways in,
/// in documented priority. The first is the repo's own convention; the other two exist because the
/// stated workflow is running this on OTHER PEOPLE'S Macs, where neither a Terminal nor a rebuild
/// is available:
///
///  1. `OMNI_PAPER=1` in the environment. Matches `OMNI_UI_DEBUG` (OmniApp.swift), `OMNI_PERF_LOG`
///     (OmniEngine.swift) and `OMNI_VALIDATED` (omni-verify). Only reaches a Terminal-launched app:
///     a Finder launch inherits launchd's environment, not the shell's.
///  2. `defaults write io.hanxiao.omni omni.paper -bool YES`. Survives a Finder launch and a
///     restart, set once per machine. New convention, and an explicit decision: it is exactly as
///     discoverable as the env var, which is to say you have to be told the key.
///  3. Holding Option while the Performance tab is on screen. The field expedient on a borrowed
///     Mac. Live-polled by the view through a flags-changed monitor, never cached, so releasing
///     Option hides the row again.
///
/// Nothing here runs at launch. Every entry point is a view body or a button action, so the paper
/// module cannot participate in startup - the v0.3.8 mistake (startup gated on GPU work, which hung
/// an M2) is structurally unavailable to it.
enum PaperGate {
    static let defaultsKey = "omni.paper"

    /// The two that persist. Cheap enough to read on every view update: an environment lookup and
    /// a `UserDefaults` bool.
    static var isPersistentlyEnabled: Bool {
        ProcessInfo.processInfo.environment["OMNI_PAPER"] == "1"
            || UserDefaults.standard.bool(forKey: defaultsKey)
    }

    /// The live modifier. `NSEvent.modifierFlags` is the current global state, not an event's, so
    /// this is correct even when the view was built before Option went down.
    static var isOptionHeld: Bool { NSEvent.modifierFlags.contains(.option) }

    static var isEnabled: Bool { isPersistentlyEnabled || isOptionHeld }
}

/// Renders `content` only while the paper gate is open, re-evaluating as Option is pressed and
/// released. The flags monitor is local (this app's events only) and is installed only while the
/// view is on screen, so nothing polls in the background.
struct PaperGated<Content: View>: View {
    @ViewBuilder var content: () -> Content
    @State private var visible = PaperGate.isEnabled
    @State private var monitor: Any?

    var body: some View {
        // The hidden branch renders a zero-size `Color.clear` rather than nothing. `Group { if
        // false { } }` collapses to an EmptyView, and SwiftUI skips lifecycle modifiers on one -
        // so `onAppear` never ran, the flags monitor was never installed, and holding Option could
        // not reveal anything. That broke the gate in EXACTLY the case it exists for: closed, on a
        // borrowed Mac, waiting for the modifier. It looked like it worked only while the
        // `omni.paper` default was set, which makes the first branch render and fires onAppear.
        Group {
            if visible { content() } else { Color.clear.frame(width: 0, height: 0) }
        }
            .onAppear {
                visible = PaperGate.isEnabled
                // SwiftUI does not guarantee balanced appear/disappear: a second onAppear would
                // overwrite the stored token and leak the first monitor, which is then unremovable
                // and keeps handling flag changes for the life of the process.
                guard monitor == nil else { return }
                monitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
                    visible = PaperGate.isPersistentlyEnabled || event.modifierFlags.contains(.option)
                    return event
                }
            }
            .onDisappear {
                if let monitor { NSEvent.removeMonitor(monitor) }
                monitor = nil
            }
    }
}
