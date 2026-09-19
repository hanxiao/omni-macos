import XCTest

/// HOLDS macOS AUTOMATION MODE OPEN SO ONE AUTHENTICATION COVERS A WHOLE RUN.
///
/// Every `xcodebuild test` that drives the UI asks testmanagerd to enable automation mode, and on
/// this OS that is a LocalAuthentication prompt - "XCTest is trying to Enable UI Automation. Enter
/// the password for the user." Nobody at the keyboard means a 60 second wait and then
///
///     Failed to initialize for UI testing: ... "Timed out while enabling automation mode."
///
/// which reads exactly like a wedged service and was diagnosed as one twice. It is not. The log
/// says what it is:
///
///     Created state file ... /var/db/com.apple.dt.automationmode/automation-enabled
///     Successfully set automation mode to ENABLED       <- after the password
///     Executing request to disable automation mode      <- when the last client goes away
///
/// The mode is a state file held open by the live test sessions, and a session that arrives while
/// it already exists does not re-authenticate. So one test that does nothing but stay alive turns
/// "a password per xcodebuild invocation" into "a password per window", which is the difference
/// between a chaos suite that can run unattended and one that cannot.
///
/// Skipped unless asked for: an idling test in the ordinary suite would be a 30 minute no-op.
final class AutomationWindowUITests: XCTestCase {

    /// Where the driver says "done, you can let go". A deadline alone would either cut a long run
    /// short or leave automation mode held long after it finished.
    static let releasePath = "/tmp/omni-automation-window.release"

    func testHoldsAutomationModeOpen() throws {
        let env = ProcessInfo.processInfo.environment
        guard let minutes = env["OMNI_AUTOMATION_HOLD_MINUTES"].flatMap(Double.init), minutes > 0 else {
            throw XCTSkip("holder only runs when OMNI_AUTOMATION_HOLD_MINUTES is set")
        }
        let fm = FileManager.default
        try? fm.removeItem(atPath: Self.releasePath)
        // stdout, not a file: the runner is sandboxed and a failed write here would be silent.
        print("[automation-window] holding for up to \(minutes) min; release at \(Self.releasePath)")
        let deadline = Date().addingTimeInterval(minutes * 60)
        while Date() < deadline {
            if fm.fileExists(atPath: Self.releasePath) {
                print("[automation-window] released after \(Int(minutes * 60 - deadline.timeIntervalSinceNow)) s")
                return
            }
            Thread.sleep(forTimeInterval: 1)
        }
        print("[automation-window] deadline reached, letting go")
    }
}
