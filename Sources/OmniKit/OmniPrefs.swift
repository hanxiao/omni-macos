import Foundation

/// Every preference Omni writes goes through here.
///
/// An isolated or UI-test run (`-omni.dbDir` or `-omni.ephemeralUIState YES` as a launch argument)
/// READS the user's settings, which the argument domain shadows, and must never WRITE them: a
/// write lands in the persistent domain whatever the argument domain says, so a test that moved a
/// slider or flipped a kind used to change the user's own app. Decided once from the launch
/// arguments, which cannot change during a session.
public enum OmniPrefs {
    public static let writesEnabled: Bool = {
        let args = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        let isolated = !((args["omni.dbDir"] as? String)?.isEmpty ?? true)
        return !isolated && !UserDefaults.standard.bool(forKey: "omni.ephemeralUIState")
    }()

    public static func set(_ value: Any?, forKey key: String) {
        if writesEnabled { UserDefaults.standard.set(value, forKey: key) }
    }

    public static func remove(_ key: String) {
        if writesEnabled { UserDefaults.standard.removeObject(forKey: key) }
    }
}
