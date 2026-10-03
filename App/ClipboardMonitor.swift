import AppKit
import OmniKit

/// Watches the general pasteboard and hands each accepted change to the clipboard history.
///
/// macOS posts no notification when the pasteboard changes, so this polls `changeCount`, which is
/// how every clipboard manager does it. Twice a second is below what a person can copy and paste
/// between, and one integer read costs nothing measurable.
///
/// What is read, in order: text (plain, or rich text as plain), then an image. Text wins when both
/// are present, because spreadsheets and word processors put an image preview beside the text a
/// person meant to copy; a bare URL beside an image loses to the image, which is what copying a
/// picture in a browser produces.
///
/// Never recorded: items marked concealed or transient (see `ClipboardHistory.skippedTypes`),
/// copied Finder files, which are already files, and the copies Omni makes itself (Copy Path, a
/// transcript, a serving token), which carry `OmniPasteboard.ownType`. What the user copies is
/// recorded whichever app is in front, Omni included.
@MainActor
final class ClipboardMonitor {
    private let history: ClipboardHistory
    private var timer: Timer?
    private var lastChange: Int
    private let queue = DispatchQueue(label: "omni.clipboard", qos: .utility)
    /// Called on the main actor after a clip is stored.
    var onStored: ((URL) -> Void)?
    /// The clip holding what is on the clipboard NOW, with its text when it is text. Nil when the
    /// current content was not recorded (concealed, a file, empty) or has not been stored yet.
    private(set) var current: (url: URL, text: String?)?

    init(history: ClipboardHistory) {
        self.history = history
        self.lastChange = NSPasteboard.general.changeCount
    }

    func start() {
        guard timer == nil else { return }
        // Whatever was on the pasteboard before capture started is not recorded, but if an earlier
        // session recorded it, that clip is the current one: pasting it must not find itself.
        lastChange = NSPasteboard.general.changeCount
        if let clip = Self.clip(from: NSPasteboard.general), let url = history.existingFile(for: clip) {
            var text: String? = nil
            if case .text(let s) = clip { text = s }
            current = (url, text)
        }
        let t = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.poll() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func poll() {
        let pb = NSPasteboard.general
        let change = pb.changeCount
        guard change != lastChange else { return }
        lastChange = change
        // Denied in System Settings > Privacy > Paste from Other Apps: reading would get nothing.
        if Self.accessDenied { return }
        current = nil
        guard let clip = Self.clip(from: pb) else { return }
        let history = history
        let text: String? = { if case .text(let s) = clip { return s }; return nil }()
        queue.async {
            guard let url = try? history.save(clip) else { return }
            Task { @MainActor [weak self] in
                guard let self, self.lastChange == change else { return }   // superseded meanwhile
                self.current = (url, text)
                self.onStored?(url)
            }
        }
    }

    /// The user set Omni to "Deny" for pasting from other apps (macOS 15.4 and later).
    static var accessDenied: Bool {
        if #available(macOS 15.4, *) { return NSPasteboard.general.accessBehavior == .alwaysDeny }
        return false
    }

    static func clip(from pb: NSPasteboard) -> ClipboardClip? {
        let types = (pb.types ?? []).map(\.rawValue)
        guard !types.isEmpty, !ClipboardHistory.shouldSkip(types: types) else { return nil }
        if types.contains(NSPasteboard.PasteboardType.fileURL.rawValue) { return nil }

        let text = pb.string(forType: .string) ?? richText(pb)
        let image = imageData(pb)
        if let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let image, isBareURL(text) { return .image(image) }
            return .text(text)
        }
        if let image { return .image(image) }
        return nil
    }

    private static func richText(_ pb: NSPasteboard) -> String? {
        if let rtf = pb.data(forType: .rtf),
           let s = NSAttributedString(rtf: rtf, documentAttributes: nil)?.string { return s }
        if let html = pb.data(forType: .html),
           let s = NSAttributedString(html: html, documentAttributes: nil)?.string { return s }
        return nil
    }

    private static func imageData(_ pb: NSPasteboard) -> Data? {
        if let png = pb.data(forType: .png) { return png }
        guard let tiff = pb.data(forType: .tiff), let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
    }

    private static func isBareURL(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.contains(where: \.isWhitespace), let u = URL(string: t) else { return false }
        return u.scheme != nil && u.host != nil
    }
}

/// Every copy Omni makes on the user's behalf goes through here, marked so the clipboard history
/// does not record Omni's own output. `concealed` also sets the nspasteboard.org concealed type,
/// which other clipboard managers honour too: the serving token is a credential.
enum OmniPasteboard {
    static let ownType = NSPasteboard.PasteboardType(ClipboardHistory.ownType)
    static let concealedType = NSPasteboard.PasteboardType("org.nspasteboard.ConcealedType")

    static func copy(_ string: String, concealed: Bool = false) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.declareTypes([.string, ownType] + (concealed ? [concealedType] : []), owner: nil)
        pb.setString(string, forType: .string)
        pb.setString("", forType: ownType)
        if concealed { pb.setString("", forType: concealedType) }
    }

    /// Files, as Finder copies them - one pasteboard item per file - with `text` on the first item
    /// for a paste into a text field. The clipboard history records NONE of this, twice over: it
    /// skips anything carrying a file URL, and the first item carries `ownType` as well. Without
    /// that, Omni copying a result would land the copy in the Clipboard folder, where it is indexed,
    /// found, and copied again. `pb` is a parameter so that guard can be checked on a private
    /// pasteboard instead of the user's clipboard.
    static func copyFiles(_ files: [URL], text: String, to pb: NSPasteboard = .general) {
        guard !files.isEmpty else { copy(text); return }
        pb.clearContents()
        let items = files.enumerated().map { i, url -> NSPasteboardItem in
            let item = NSPasteboardItem()
            item.setString(url.absoluteString, forType: .fileURL)
            if i == 0 {
                item.setString(text, forType: .string)
                item.setString("", forType: ownType)
            }
            return item
        }
        pb.writeObjects(items)
    }
}

/// Confirmation for deleting the whole clipboard history, shared by the sidebar and Settings.
@MainActor
enum ClipboardClear {
    static func confirm(_ model: AppModel) {
        let n = model.clipboardClipCount
        let alert = NSAlert()
        alert.messageText = "Clear clipboard history?"
        alert.informativeText = n == 1 ? "1 clip will be deleted." : "\(n.formatted()) clips will be deleted."
        alert.addButton(withTitle: "Clear")
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { model.clearClipboardHistory() }
    }
}
