import SwiftUI

/// Shared spacing/radius tokens so paddings and corners stay on a consistent rhythm.
enum Design {
    static let corner: CGFloat = 8     // cards, thumbnails, selection halos
    static let cornerSmall: CGFloat = 6
    static let gap: CGFloat = 8
    static let gapLarge: CGFloat = 16
}

extension View {
    /// A small translucent chip for labels floating over imagery (e.g. the relevance badge over a
    /// thumbnail). Uses native Liquid Glass on macOS 26 - Apple's guidance is to prefer the
    /// `glassEffect` API over a hand-rolled blur - and falls back to a material capsule on 14-15.
    /// Non-interactive by design: the chip is a passive label, so no `.interactive()`.
    @ViewBuilder func glassChip() -> some View {
        if #available(macOS 26, *) {
            self.glassEffect(.regular, in: .capsule)
        } else {
            self.background(.ultraThinMaterial, in: Capsule())
        }
    }
}
