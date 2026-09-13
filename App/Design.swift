import SwiftUI
import AppKit
import OmniKit

/// Shared spacing/radius tokens so paddings and corners stay on a consistent rhythm.
enum Design {
    static let corner: CGFloat = 8     // cards, thumbnails, selection halos
    static let cornerSmall: CGFloat = 6
    static let gap: CGFloat = 8
    static let gapLarge: CGFloat = 16
}

/// File and memory sizes, spelled the way the system spells them.
///
/// Two reasons this is not `ByteCountFormatter.string(fromByteCount:countStyle:)` at each call
/// site. The static convenience leaves `allowsNonnumericFormatting` ON, which renders 0 as
/// "Zero KB" - that shipped in the memory legend, and no Apple surface says it; Finder's Get Info
/// says "0 bytes". And `Int.formatted(.byteCount(style:))`, the modern spelling, renders SI case
/// ("454 kB") where Finder writes "454 KB".
/// `@MainActor` because `ByteCountFormatter` is not `Sendable` and these instances are cached
/// rather than rebuilt per row - a directory listing formats one size per visible row. Every call
/// site is already main-actor (view bodies, and the repair message inside a `MainActor.run`).
@MainActor
enum ByteSize {
    private static func formatter(_ style: ByteCountFormatter.CountStyle) -> ByteCountFormatter {
        let f = ByteCountFormatter()
        f.countStyle = style
        f.allowsNonnumericFormatting = false
        return f
    }
    private static let fileFormatter = formatter(.file)
    private static let memoryFormatter = formatter(.memory)

    /// Disk and document sizes - what Finder shows in a Size column.
    static func file(_ bytes: some BinaryInteger) -> String {
        fileFormatter.string(fromByteCount: Int64(bytes))
    }

    /// Resident memory, which the system reports in powers of two.
    static func memory(_ bytes: some BinaryInteger) -> String {
        memoryFormatter.string(fromByteCount: Int64(bytes))
    }
}

/// Applies a Liquid Glass capsule, but honors Reduce Transparency: when the user has it on (an
/// accessibility preference, also common on low-power/older Macs), it drops the live vibrancy sample
/// for a flat material capsule. That is the HIG-correct behavior AND removes a per-chip glass pass -
/// material on the same grid of N visible cells (or the map's overlays over the live Metal cloud)
/// costs far less GPU than N live `glassEffect` recomposites per frame.
private struct GlassChip: ViewModifier {
    var interactive: Bool
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    func body(content: Content) -> some View {
        if #available(macOS 26, *), !reduceTransparency {
            content.glassEffect(interactive ? .regular.interactive() : .regular, in: .capsule)
        } else {
            content.background(.ultraThinMaterial, in: Capsule())
        }
    }
}

extension View {
    /// A small translucent chip for content floating over imagery. Uses native Liquid Glass on
    /// macOS 26 - Apple's guidance is to prefer the `glassEffect` API over a hand-rolled blur - and
    /// falls back to a material capsule on 14-15 or when Reduce Transparency is on. Pass
    /// `interactive: true` for a chip that hosts controls (e.g. the map's zoom cluster) so the glass
    /// responds to the pointer; leave it false for passive labels/badges.
    func glassChip(interactive: Bool = false) -> some View {
        modifier(GlassChip(interactive: interactive))
    }
}

/// Groups sibling Liquid Glass surfaces into one `GlassEffectContainer` on macOS 26 - Apple's
/// requirement when several glass elements coexist: the container renders them in a single
/// effect pass (cheaper than N independent passes) and lets effects that approach each other
/// blend instead of stacking. On macOS 14-15 (material fallback chips) it is a no-op wrapper.
/// `spacing` is the effect-merge distance, not layout spacing.
struct GlassGroup<Content: View>: View {
    var spacing: CGFloat = 12
    @ViewBuilder var content: Content
    var body: some View {
        if #available(macOS 26, *) {
            GlassEffectContainer(spacing: spacing) { content }
        } else {
            content
        }
    }
}

extension FileKind {
    /// The dynamic system color for this kind. NSColor's system colors ship distinct light and dark
    /// variants (blue #007AFF/#0A84FF, purple #AF52DE/#BF5AF2, etc.) and adapt to the active
    /// appearance automatically - never hand-inverted, per the macOS HIG. The folder-map legend and
    /// per-extension dot shades both derive from this so the palette stays correct in both modes.
    var vizNSColor: NSColor {
        switch self {
        case .image: return .systemBlue
        case .video: return .systemPurple
        case .audio: return .systemOrange
        case .text:  return .systemGreen
        case .scan:  return .systemBrown   // manila-paper hue, distinct from the four above
        }
    }

    /// Main hue per kind: the legend swatch and the "one color per type". Fully appearance-adaptive.
    var vizColor: Color { Color(nsColor: vizNSColor) }

    /// Per-extension shade within the kind's hue family, as straight RGBA for the GPU point buffer.
    /// `base` is the kind's HSB already resolved for the active appearance (see
    /// FolderEmbeddingVisualization.rebuildColors); the kind keeps its hue (its identity) and the
    /// extension only nudges saturation/brightness, so .md/.txt/.pdf are distinguishable greens,
    /// .png/.jpg distinguishable blues, while the kind stays obvious. `alpha` carries the dot density
    /// alpha. HSB->RGB is done inline (no per-point NSColor allocation - this runs over every file).
    static func vizShadeRGBA(base: (h: CGFloat, s: CGFloat, b: CGFloat), ext: String, alpha: Float) -> SIMD4<Float> {
        var h = Float(base.h), s = Float(base.s), b = Float(base.b)
        let e = ext.lowercased()
        if !e.isEmpty {
            var hash: UInt64 = 1469598103934665603             // FNV-1a -> stable factor per extension
            for byte in e.utf8 { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
            let t = Float(hash % 997) / 997.0
            h = (h + (t - 0.5) * 0.04 + 1).truncatingRemainder(dividingBy: 1)   // tiny jitter, kept in family
            s = min(1, max(0.55, s * (0.80 + 0.28 * t)))
            b = min(1, max(0.62, b * (0.84 + 0.24 * (1 - t))))
        }
        let (r, g, bl) = Self.hsb2rgb(h, s, b)
        return SIMD4<Float>(r, g, bl, alpha)
    }

    private static func hsb2rgb(_ h: Float, _ s: Float, _ b: Float) -> (Float, Float, Float) {
        if s <= 0 { return (b, b, b) }
        let h6 = (h - h.rounded(.down)) * 6
        let i = Int(h6), f = h6 - Float(i)
        let p = b * (1 - s), q = b * (1 - s * f), t = b * (1 - s * (1 - f))
        switch i % 6 {
        case 0: return (b, t, p)
        case 1: return (q, b, p)
        case 2: return (p, b, t)
        case 3: return (p, q, b)
        case 4: return (t, p, b)
        default: return (b, p, q)
        }
    }
}

/// A toolbar button that is either on or off, drawn the way the system draws one.
///
/// ONE STYLE FOR EVERY SUCH BUTTON in this app. The OCR toggle used to hand-draw its on state - an
/// accent `Circle` behind a white glyph, painted inside the label - which had to be re-derived for
/// the next toggle that came along, and was a guess at the platform's look rather than the platform's
/// look. `Toggle` + `.toggleStyle(.button)` is the control the system provides for exactly this: on
/// Tahoe the item's own capsule fills with the accent colour and the glyph inverts, which is what
/// Finder's toolbar toggles do, and it stays correct through theme, accent-colour and appearance
/// changes without this app knowing about any of them.
struct ToolbarToggle: View {
    @Binding var isOn: Bool
    let symbol: String

    var body: some View {
        Toggle(isOn: $isOn) { Image(systemName: symbol) }
            .toggleStyle(.button)
    }
}
