import SwiftUI
import AppKit
import UniformTypeIdentifiers
import OmniKit

/// The paper run's final dialog: what was measured, on what machine, and what about the run makes
/// it suspect - shown before the operator sends the file, not after.
///
/// Modelled on AgentConfigSheet (Serving/ServingTab.swift): the same selectable monospaced
/// ScrollView on `.textBackgroundColor`, the same Copy-with-label-flip, the same NSSavePanel and
/// `.onExitCommand`. The body is exactly the text that gets saved, character for character - there
/// is no second rendering path that could disagree with the file.
struct PaperResultSheet: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let report = model.lastPaperReport {
                header(report)
                let warnings = Self.warnings(report.result)
                if !warnings.isEmpty { warningStrip(warnings) }
                reportBox(model.lastPaperReportText)
                footer(report)
            } else {
                // Only reachable if the report was cleared underneath the sheet. Say so rather than
                // showing an empty box that looks like a run which measured nothing.
                Text("No paper report").font(.headline)
                Text("The run produced no report to show.")
                    .font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                HStack { Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.defaultAction) }
            }
        }
        .padding(20)
        .frame(width: 720, height: 560)
        .onExitCommand { dismiss() }   // Esc closes, matching the native sheet expectation
    }

    // MARK: - Pieces

    @ViewBuilder private func header(_ report: PaperReport) -> some View {
        Text("Paper benchmark \u{00B7} \(report.result.status.rawValue)").font(.headline)
        Text(Self.subtitle(report))
            .font(.subheadline).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// Capped at six lines: a run where every case skipped would otherwise push the report itself
    /// off the sheet, and the full list is in the export's own warning block either way.
    @ViewBuilder private func warningStrip(_ warnings: [String]) -> some View {
        let shown = warnings.prefix(6)
        VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(shown), id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .labelStyle(.titleAndIcon)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if warnings.count > shown.count {
                Text("+\(warnings.count - shown.count) more, listed in the report below")
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .foregroundStyle(.orange)
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder private func reportBox(_ text: String) -> some View {
        ScrollView {
            Text(text)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
        }
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.primary.opacity(0.1)))
    }

    @ViewBuilder private func footer(_ report: PaperReport) -> some View {
        HStack {
            Button(copied ? "Copied" : "Copy") {
                let pb = NSPasteboard.general
                pb.clearContents()
                pb.setString(model.lastPaperReportText, forType: .string)
                copied = true
                Task { try? await Task.sleep(for: .seconds(1.5)); copied = false }
            }
            Button("Save\u{2026}") {
                let panel = NSSavePanel()
                panel.nameFieldStringValue = report.suggestedFileName
                panel.allowedContentTypes = [.plainText]
                if panel.runModal() == .OK, let url = panel.url {
                    try? model.lastPaperReportText.write(to: url, atomically: true, encoding: .utf8)
                }
            }
            Spacer()
            Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
        }
        // The auto-saved copy, written before this dialog appeared. Stated so that Done, a crash or
        // a closed laptop is never what loses a run that took up to 25 minutes.
        if let url = model.lastPaperReportURL {
            Text("Also saved to \(url.path)")
                .font(.caption).foregroundStyle(.secondary)
                .textSelection(.enabled)
                .lineLimit(1).truncationMode(.middle)
        } else {
            Text("Could not auto-save a copy - use Save or Copy before closing.")
                .font(.caption).foregroundStyle(.orange)
        }
    }

    // MARK: - Lines

    /// "Apple M2 (4P+4E) · 8 GB · CAP-3 · nano · suite paper-v1 · 9m 12s · 11 of 11 cases ok"
    private static func subtitle(_ report: PaperReport) -> String {
        let s = report.statics
        let cores = s.cpuECores > 0 ? " (\(s.cpuPCores)P+\(s.cpuECores)E)" : ""
        let memory = String(format: "%.0f GB", Double(s.memoryBytes) / 1_073_741_824)
        let r = report.result
        return [s.chip + cores, memory, r.capClass.rawValue, report.app.modelVariant,
                "suite " + r.suite, duration(r.wallSeconds),
                "\(r.casesOK) of \(r.casesTotal) cases ok"].joined(separator: "  \u{00B7}  ")
    }

    /// What makes this run suspect. Derived from the result, never from the rendered text: the
    /// export carries the same warnings, and both must come from the same numbers.
    static func warnings(_ r: PaperSuiteResult) -> [String] {
        var w: [String] = []
        if r.status != .complete {
            w.append("PARTIAL RUN (\(r.status.rawValue), \(r.casesOK) of \(r.casesTotal) cases ok). "
                     + "Do not present this as a complete suite.")
        }
        if r.scale < 1.0 {
            w.append(String(format: "Smoke run at scale %.3f: every rate covers less work than the paper's.", r.scale))
        }
        for c in r.cases where c.status != .ok {
            w.append("\(c.id) \(c.status.rawValue)" + (c.note.map { ": " + $0 } ?? ""))
        }
        if r.thermalDriftExceeded, let drift = r.thermalDriftPercent {
            w.append(String(format: "Thermal drift %.1f%% exceeds +-8%%: the machine changed clock domain mid-suite.", drift))
        }
        if r.maxThermal == "serious" || r.maxThermal == "critical" {
            w.append("Thermal state reached '\(r.maxThermal)': the machine throttled during the run.")
        }
        if r.swapDeltaMB > 256 {
            w.append(String(format: "Swap grew by %.0f MB: part of this run was a paging measurement.", r.swapDeltaMB))
        }
        if !r.contendedCases.isEmpty {
            w.append("Contended (another process used more than a fifth of the machine): "
                     + r.contendedCases.joined(separator: ", "))
        }
        return w
    }

    private static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s < 60 ? "\(s)s" : "\(s / 60)m \(s % 60)s"
    }
}
