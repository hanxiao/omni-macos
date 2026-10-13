import AppKit
import Network
import OmniKit
import OmniLink
import SwiftUI

// MARK: - The Link a Mac window

/// Pairing (docs/notes/link.md, "Pairing"), laid out like Finder's Connect to Server: a list,
/// an address field, Cancel and a default button. Open on BOTH Macs: while it is open this Mac is
/// visible and accepts a request. The Mac that starts shows six digits; the other asks for them.
struct LinkSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var address = ""
    @State private var picked: String?
    @State private var typed = ""
    @State private var typedError: String?
    @FocusState private var codeFocused: Bool

    private var link: LinkController { model.link }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let req = link.incoming {
                incoming(req)
            } else if let out = link.outgoing {
                outgoing(out)
            } else if let msg = link.incomingMessage {
                Text(msg).font(.headline)
                buttons { Button("Done") { dismiss() }.keyboardShortcut(.defaultAction) }
            } else {
                choose
            }
        }
        .padding(20)
        .frame(width: 420)
        .onAppear { link.setWindowOpen(true) }
        .onDisappear { link.setWindowOpen(false) }
    }

    private func buttons(@ViewBuilder _ content: () -> some View) -> some View {
        HStack(spacing: 8) { Spacer(); content() }.padding(.top, 4)
    }

    // Pick a Mac nearby, or type an address.
    @ViewBuilder private var choose: some View {
        @Bindable var link = model.link
        Text("Link a Mac").font(.headline)
        List(link.available, selection: $picked) { mac in
            Label(mac.name, systemImage: "desktopcomputer").tag(mac.id)
        }
        .listStyle(.bordered)
        .alternatingRowBackgrounds()
        .frame(height: 132)
        .overlay {
            if link.available.isEmpty {
                VStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Open Link a Mac on the other Mac").foregroundStyle(.secondary)
                }
                .font(.callout)
            }
        }
        .onChange(of: picked) { _, p in if p != nil { address = "" } }
        TextField("Name or address", text: $address)
            .textFieldStyle(.roundedBorder)
            .onChange(of: address) { _, a in if !a.isEmpty { picked = nil } }
            .onSubmit(start)
        Toggle("Allow linked Macs to search this Mac", isOn: $link.alsoShare)
        HStack(spacing: 8) {
            Text(link.port.map { "\(link.name) \u{00B7} port \(String($0))" } ?? link.name)
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
            Button("Link", action: start).keyboardShortcut(.defaultAction)
                .disabled(picked == nil && address.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.top, 4)
    }

    private func start() {
        if let id = picked, let mac = link.available.first(where: { $0.id == id }) {
            link.pair(mac.endpoint, name: mac.name)
        } else if !address.trimmingCharacters(in: .whitespaces).isEmpty {
            link.pair(address: address)
        }
    }

    // This Mac asked: show the code, or how it went.
    @ViewBuilder private func outgoing(_ out: LinkController.Outgoing) -> some View {
        if out.linked {
            Label("Linked with \(out.name)", systemImage: "checkmark.circle.fill")
                .font(.headline).symbolRenderingMode(.multicolor)
            buttons {
                Button("Link Another") { link.cancelPairing() }
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        } else if let err = out.error {
            Text("Couldn\u{2019}t link with \(out.name)").font(.headline)
            Text(err).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            buttons {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Back") { link.cancelPairing() }.keyboardShortcut(.defaultAction)
            }
        } else if let code = out.code {
            Text("Enter this code on \(out.name)").font(.headline)
            Text(LinkPairing.display(code))
                .font(.system(size: 40, weight: .semibold, design: .rounded).monospacedDigit())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .accessibilityIdentifier("link.shownCode")
            buttons { Button("Cancel") { link.cancelPairing() }.keyboardShortcut(.cancelAction) }
        } else {
            Text("Link a Mac").font(.headline)
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Connecting to \(out.name)\u{2026}").foregroundStyle(.secondary)
            }
            buttons { Button("Cancel") { link.cancelPairing() }.keyboardShortcut(.cancelAction) }
        }
    }

    // The other Mac asked; type what it shows.
    @ViewBuilder private func incoming(_ req: IncomingPairing) -> some View {
        Text("Link with \u{201C}\(req.name)\u{201D}?").font(.headline)
        Text("Enter the code shown on \(req.name).").foregroundStyle(.secondary)
        TextField("000 000", text: $typed)
            .font(.system(size: 28, weight: .semibold, design: .rounded).monospacedDigit())
            .multilineTextAlignment(.center)
            .textFieldStyle(.roundedBorder)
            .focused($codeFocused)
            .onSubmit(submit)
            .accessibilityIdentifier("link.code")
            .onAppear { typed = ""; typedError = nil; codeFocused = true }
        if let typedError { Text(typedError).font(.callout).foregroundStyle(.red) }
        buttons {
            Button("Decline") { link.declineIncoming(); typed = "" }.keyboardShortcut(.cancelAction)
            Button("Link", action: submit).keyboardShortcut(.defaultAction)
                .disabled(LinkPairing.normalize(typed).count != 6)
        }
    }

    private func submit() {
        typedError = link.submit(typed)
        if typedError != nil { typed = "" }
    }
}

// MARK: - Browsing a linked Mac

/// A linked Mac, or a folder on it, when the box is empty: its sources at the top, then folders
/// and files, read from that Mac's index over the link. The counterpart of FolderBrowser.
struct RemoteBrowser: View {
    @Environment(AppModel.self) private var model
    let scope: AppModel.LinkedScope

    @State private var children: [LinkWire.Child] = []
    @State private var loading = true
    @State private var failure: String?
    @State private var selected: String?

    private var link: LinkController { model.link }
    private var macName: String { link.name(of: scope.peer) }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !link.isOnline(scope.peer) {
                CenteredStatus(symbol: CenteredStatus.moleSleep, title: "\(macName) is offline", subtitle: "",
                               showSpinner: false, action: ("Try Again", { link.retry(scope.peer) }))
            } else if link.peer(scope.peer)?.theyAllowSearch == false {
                CenteredStatus(symbol: CenteredStatus.moleSad, title: "\(macName) isn\u{2019}t sharing with this Mac",
                               subtitle: "", showSpinner: false)
            } else if let failure {
                CenteredStatus(symbol: CenteredStatus.moleSad, title: failure, subtitle: "", showSpinner: false)
            } else if loading && children.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if children.isEmpty {
                CenteredStatus(symbol: CenteredStatus.moleEmpty, title: "Nothing indexed here", subtitle: "", showSpinner: false)
            } else {
                list
            }
        }
        .task(id: "\(scope.peer)|\(scope.folder ?? "")|\(link.isOnline(scope.peer))") { await load() }
        .quickLookPreview(Binding(get: { model.previewURL },
                                  set: { if $0 != model.previewURL { model.previewURL = $0 } }))
    }

    private func load() async {
        loading = true
        failure = nil
        defer { loading = false }
        guard link.isOnline(scope.peer) else { children = []; return }
        if let folder = scope.folder {
            do {
                let kids = try await link.list(scope.peer, folder: folder)
                children = kids.sorted { a, b in
                    a.folder != b.folder ? a.folder
                        : (a.path as NSString).lastPathComponent.localizedStandardCompare((b.path as NSString).lastPathComponent) == .orderedAscending
                }
            } catch {
                failure = LinkController.describe(error)
            }
        } else {
            await link.reloadSources(scope.peer)
            children = (link.sources[scope.peer] ?? []).map {
                LinkWire.Child(path: $0.key, folder: true, kind: $0.kind, modified: 0, size: 0, files: $0.files)
            }
        }
    }

    /// The browsers' column header: the location is in the scope chip above, as a folder's name is
    /// in the toolbar for FolderBrowser.
    private var header: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            HStack(spacing: 6) {
                Text("Name").foregroundStyle(.secondary)
                if !children.isEmpty {
                    Text("\(children.count.formatted()) item\(children.count == 1 ? "" : "s")").foregroundStyle(.tertiary)
                }
                Spacer(minLength: 0)
            }
            Text("Date Modified").foregroundStyle(.secondary).lineLimit(1)
                .modifier(ColumnSlot(col: .dateModified, alignment: .leading))
                .overlay(alignment: .leading) { Divider().frame(height: 12) }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, BrowserMetrics.rowTrail + BrowserMetrics.listInset)
        .padding(.vertical, 5)
    }

    private func displayName(_ path: String) -> String {
        if let s = link.sources[scope.peer]?.first(where: { $0.key == path }) { return s.name }
        return (path as NSString).lastPathComponent
    }

    private func remote(_ c: LinkWire.Child) -> String { RemotePath.make(peer: scope.peer, path: c.path) }

    private var list: some View {
        List(children, id: \.path, selection: $selected) { c in
            let isSelected = selected == c.path
            HStack(spacing: 0) {
                Group {
                    if c.folder {
                        Image(systemName: c.kind == "photos" ? "photo.on.rectangle.angled" : "folder.fill")
                            .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(Color.accentColor))
                            .frame(width: BrowserMetrics.icon, height: BrowserMetrics.icon)
                    } else {
                        Thumbnail(path: remote(c), side: BrowserMetrics.icon, corner: 3)
                    }
                }
                .padding(.trailing, BrowserMetrics.iconGap)
                Text(c.folder ? displayName(c.path) : (c.path as NSString).lastPathComponent)
                    .lineLimit(1).truncationMode(.middle)
                    .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                Spacer(minLength: 8)
                Text(c.modified > 0 ? Date(timeIntervalSince1970: c.modified).formatted(date: .abbreviated, time: .shortened) : "")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                    .lineLimit(1)
                    .modifier(ColumnSlot(col: .dateModified, alignment: .leading))
            }
            .padding(.leading, BrowserMetrics.rowLead)
            .padding(.trailing, BrowserMetrics.rowTrail)
            .contentShape(RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
            .onTapGesture(count: 2) { activate(c) }
            .simultaneousGesture(TapGesture().onEnded { select(c) })
            .modifier(RemoteFileMenu(child: c, path: remote(c)))
            .listRowSeparator(.hidden)
            .listRowBackground(
                RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius)
                    .fill(isSelected ? Color.accentColor : Color.clear)
                    .padding(.horizontal, BrowserMetrics.selectionInset)
            )
            .listRowInsets(EdgeInsets())
        }
        .environment(\.defaultMinListRowHeight, BrowserMetrics.rowHeight)
        .modifier(SoftTopScrollEdge())
        .alternatingRowBackgrounds()
        .listStyle(.inset)
    }

    private func select(_ c: LinkWire.Child) {
        selected = c.path
        if !c.folder { model.selectSingle(remote(c)) }
    }

    private func activate(_ c: LinkWire.Child) {
        if c.folder { model.enterLinkedMac(scope.peer, folder: c.path) } else { model.openResult(remote(c)) }
    }
}

/// The file menu on a linked Mac's file in the browser; a folder offers to search inside it.
private struct RemoteFileMenu: ViewModifier {
    @Environment(AppModel.self) private var model
    let child: LinkWire.Child
    let path: String
    func body(content: Content) -> some View {
        content.contextMenu {
            if child.folder {
                Button { model.enterLinkedMac(RemotePath.peer(of: path) ?? "", folder: child.path) } label: {
                    Label("Open", systemImage: "arrow.up.forward.app")
                }
                Button {
                    model.enterLinkedMac(RemotePath.peer(of: path) ?? "", folder: child.path)
                    SearchFieldFocus.focus()
                } label: { Label("Search in This Folder", systemImage: "magnifyingglass") }
            } else {
                FileMenuItems(path: path, kind: child.kind)
            }
        }
    }
}

// MARK: - Scope chip and fetch status

/// Where a search or a listing on a linked Mac is: "Studio › Documents › Taxes", each part a step
/// back up, the way a path bar is. The x leaves the linked Mac for this Mac and every linked Mac.
struct LinkedScopeChip: View {
    @Environment(AppModel.self) private var model
    let scope: AppModel.LinkedScope

    /// (title, folder to go to) from the Mac down to the folder on screen.
    private var crumbs: [(String, String?)] {
        let link = model.link
        var out: [(String, String?)] = [(link.name(of: scope.peer), nil)]
        guard let f = scope.folder else { return out }
        if let src = link.sources[scope.peer]?.first(where: { f == $0.key || f.hasPrefix($0.key + "/") }) {
            out.append((src.name, src.key))
            var path = src.key
            for part in f.dropFirst(src.key.count).split(separator: "/") {
                path += "/" + part
                out.append((String(part), path))
            }
        } else {
            out.append(((f as NSString).lastPathComponent, f))
        }
        return out
    }

    var body: some View {
        let online = model.link.isOnline(scope.peer)
        HStack(spacing: 6) {
            Image(systemName: "desktopcomputer").foregroundStyle(.secondary).frame(width: 18)
            ForEach(Array(crumbs.enumerated()), id: \.offset) { i, c in
                if i > 0 { Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary) }
                Button(c.0) { model.enterLinkedMac(scope.peer, folder: c.1) }
                    .buttonStyle(.plain)
                    .fontWeight(i == crumbs.count - 1 ? .medium : .regular)
                    .foregroundStyle(online ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            Button { model.leaveLinkedScope() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary).help("Search Everywhere")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 8)
    }
}

/// "Downloading report.pdf" while a linked Mac's file is fetched to open or preview.
struct RemoteFetchStatus: View {
    @Environment(AppModel.self) private var model
    @State private var fraction: Double = 0
    var body: some View {
        if let path = model.remoteOpening {
            HStack(spacing: 10) {
                ProgressView(value: fraction).frame(width: 120)
                Text("Downloading \u{201C}\((path as NSString).lastPathComponent)\u{201D}")
                    .lineLimit(1).truncationMode(.middle)
                Button("Cancel") { model.remoteOpening = nil }.buttonStyle(.borderless)
            }
            .font(.callout)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
            .padding(.bottom, 14)
            .task(id: path) {
                while !Task.isCancelled {
                    fraction = RemoteFiles.shared.fraction(path) ?? fraction
                    try? await Task.sleep(for: .milliseconds(150))
                }
            }
        }
    }
}

enum LinkedMacIcon {
    static func symbol(_ model: String) -> String {
        model.contains("Book") ? "laptopcomputer" : "desktopcomputer"
    }

    @MainActor static func status(_ link: LinkController, _ p: LinkPeer) -> String {
        switch link.state(p.id) {
        case .online: return p.theyAllowSearch ? "Connected" : "Not sharing with this Mac"
        case .connecting: return "Connecting\u{2026}"
        case .offline: return "Offline"
        case .unlinked: return "Unlinked on \(p.name)"
        }
    }
}
