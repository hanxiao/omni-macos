import AppKit
import SwiftUI

/// LaTeX math from transcribed documents, set natively.
///
/// The model is asked to emit `$…$` and `$$…$$`, and until now the formatted view showed those
/// dollars and backslashes verbatim - the one part of a page that came out looking worse than the
/// scan it came from.
///
/// There is no system LaTeX engine on macOS. The usual answers are a WebView running KaTeX or a
/// third-party typesetter: a browser and a network round trip, or a dependency, to set an inline
/// fraction. What document OCR actually produces is not journal typesetting - units, indices,
/// Greek letters, simple fractions and roots - and that maps onto Unicode and `AttributedString`'s
/// own baseline offsets, which `Text` lays out, selects and copies like any other text.
///
/// What it deliberately does NOT do: matrices, integrals with limits set above and below, nested
/// fractions. Those come out in a readable linear form (`a⁄b`, `∫_0^1`) rather than being silently
/// mis-set, and a command it does not know keeps its backslash so it reads as source rather than
/// as a claim.
enum MathText {

    /// Split a line into text and `$…$` math, rendering each. Math is found BEFORE the Markdown
    /// parser runs: `$x^2$` contains `^` and `_`, which the inline parser would otherwise treat as
    /// emphasis and eat.
    static func inline(_ source: String, text: (String) -> AttributedString) -> AttributedString {
        guard source.contains("$") else { return text(source) }
        var out = AttributedString()
        var rest = Substring(source)
        while let open = rest.firstIndex(of: "$") {
            let before = rest[rest.startIndex ..< open]
            let afterOpen = rest.index(after: open)
            // `$$` inline is display math on one line; treat it as a longer delimiter.
            let double = afterOpen < rest.endIndex && rest[afterOpen] == "$"
            let bodyStart = double ? rest.index(after: afterOpen) : afterOpen
            let close = double
                ? rest.range(of: "$$", range: bodyStart ..< rest.endIndex)?.lowerBound
                : rest[bodyStart...].firstIndex(of: "$")
            guard let close else { break }        // an unclosed `$` is a dollar sign, not math
            if !before.isEmpty { out.append(text(String(before))) }
            out.append(render(String(rest[bodyStart ..< close])))
            rest = rest[(double ? rest.index(close, offsetBy: 2) : rest.index(after: close))...]
        }
        if !rest.isEmpty { out.append(text(String(rest))) }
        return out
    }

    /// One math expression.
    static func render(_ latex: String) -> AttributedString {
        var scanner = Scanner(latex)
        return scanner.run()
    }

    // MARK: - Scanning

    private struct Scanner {
        private let chars: [Character]
        private var i = 0
        init(_ s: String) { chars = Array(s) }

        mutating func run(stopAtBrace: Bool = false) -> AttributedString {
            var out = AttributedString()
            while i < chars.count {
                let c = chars[i]
                if c == "}" , stopAtBrace { break }
                switch c {
                case "\\":
                    out.append(command())
                case "^":
                    i += 1
                    out.append(script(group(), superscript: true))
                case "_":
                    i += 1
                    out.append(script(group(), superscript: false))
                case "{":
                    i += 1
                    out.append(run(stopAtBrace: true))
                    if i < chars.count, chars[i] == "}" { i += 1 }
                case "}":
                    i += 1
                case " ":
                    // LaTeX collapses runs of spaces; keeping them would space an expression out.
                    while i < chars.count, chars[i] == " " { i += 1 }
                    if !out.characters.isEmpty { out.append(AttributedString(" ")) }
                default:
                    i += 1
                    out.append(literal(String(c)))
                }
            }
            return out
        }

        /// A `{…}` group, or the single token that follows when there are no braces (`x^2`).
        private mutating func group() -> AttributedString {
            guard i < chars.count else { return AttributedString() }
            if chars[i] == "{" {
                i += 1
                let inner = run(stopAtBrace: true)
                if i < chars.count, chars[i] == "}" { i += 1 }
                return inner
            }
            if chars[i] == "\\" { return command() }
            let c = chars[i]; i += 1
            return literal(String(c))
        }

        private mutating func command() -> AttributedString {
            i += 1                                    // the backslash
            guard i < chars.count else { return AttributedString("\\") }
            // Escaped punctuation: \$ \% \& \_ \{ \}
            if !chars[i].isLetter {
                let c = chars[i]; i += 1
                return AttributedString(String(c))
            }
            var name = ""
            while i < chars.count, chars[i].isLetter { name.append(chars[i]); i += 1 }
            switch name {
            case "frac", "dfrac", "tfrac":
                let numerator = group(), denominator = group()
                return fraction(numerator, denominator)
            case "sqrt":
                var body = group()
                if body.characters.count > 1 {
                    body = AttributedString("(") + body + AttributedString(")")
                }
                return AttributedString("\u{221A}") + body
            case "text", "mathrm", "mathbf", "operatorname", "mathit", "mathsf":
                var body = group()
                // Upright, because that is what these commands are for.
                body.appKit.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
                if name == "mathbf" {
                    body.appKit.font = NSFont.boldSystemFont(ofSize: NSFont.systemFontSize)
                }
                return body
            case "left", "right", "bigl", "bigr", "Bigl", "Bigr":
                return AttributedString()             // the delimiter itself follows as a literal
            case "hat", "bar", "vec", "tilde", "dot", "ddot", "overline", "widehat", "acute", "grave":
                // Accents are combining marks, which is how Unicode sets them and how they copy.
                let mark = MathText.accents[name].map(String.init) ?? ""
                return group() + AttributedString(mark)
            case "quad": return AttributedString("\u{2003}")
            case "qquad": return AttributedString("\u{2003}\u{2003}")
            default:
                if let symbol = MathText.symbols[name] { return AttributedString(symbol) }
                if MathText.functions.contains(name) {
                    var f = AttributedString(name)
                    f.appKit.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
                    return f
                }
                // Not known: keep the backslash so it reads as source rather than as a claim,
                // and a trailing space so it cannot fuse with whatever follows into a word that
                // looks like a command nobody wrote.
                return AttributedString("\\" + name + " ")
            }
        }
    }

    // MARK: - Pieces

    /// Variables are italic and everything else upright, which is the one typographic convention
    /// that makes maths readable as maths rather than as a string of letters.
    private static func literal(_ s: String) -> AttributedString {
        var out = AttributedString(s)
        let size = NSFont.systemFontSize
        if let c = s.first, c.isLetter {
            out.appKit.font = NSFont(descriptor: NSFont.systemFont(ofSize: size).fontDescriptor
                .withSymbolicTraits(.italic), size: size)
                ?? NSFont.systemFont(ofSize: size)
        }
        return out
    }

    /// Unicode where the whole run maps cleanly - it aligns and copies far better than a shifted
    /// baseline - and a raised smaller run otherwise.
    private static func script(_ body: AttributedString, superscript: Bool) -> AttributedString {
        let plain = String(body.characters)
        let table = superscript ? superscripts : subscripts
        if !plain.isEmpty, plain.allSatisfy({ table[$0] != nil }) {
            return AttributedString(String(plain.map { table[$0]! }))
        }
        var out = body
        let size = NSFont.systemFontSize * 0.72
        out.appKit.font = NSFont.systemFont(ofSize: size)
        out.baselineOffset = superscript ? size * 0.45 : -size * 0.22
        return out
    }

    private static func fraction(_ numerator: AttributedString,
                                 _ denominator: AttributedString) -> AttributedString {
        func wrapped(_ part: AttributedString) -> AttributedString {
            part.characters.count > 1
                ? AttributedString("(") + part + AttributedString(")")
                : part
        }
        // U+2044 FRACTION SLASH, which fonts kern as a fraction rather than as division.
        return wrapped(numerator) + AttributedString("\u{2044}") + wrapped(denominator)
    }

    /// Combining marks, applied AFTER the glyph they sit on.
    private static let accents: [String: Character] = [
        "hat": "\u{0302}", "widehat": "\u{0302}", "bar": "\u{0304}", "overline": "\u{0304}",
        "vec": "\u{20D7}", "tilde": "\u{0303}", "dot": "\u{0307}", "ddot": "\u{0308}",
        "acute": "\u{0301}", "grave": "\u{0300}",
    ]

    private static let superscripts: [Character: Character] = [
        "0": "\u{2070}", "1": "\u{00B9}", "2": "\u{00B2}", "3": "\u{00B3}", "4": "\u{2074}",
        "5": "\u{2075}", "6": "\u{2076}", "7": "\u{2077}", "8": "\u{2078}", "9": "\u{2079}",
        "+": "\u{207A}", "-": "\u{207B}", "=": "\u{207C}", "(": "\u{207D}", ")": "\u{207E}",
        "n": "\u{207F}", "i": "\u{2071}",
    ]

    private static let subscripts: [Character: Character] = [
        "0": "\u{2080}", "1": "\u{2081}", "2": "\u{2082}", "3": "\u{2083}", "4": "\u{2084}",
        "5": "\u{2085}", "6": "\u{2086}", "7": "\u{2087}", "8": "\u{2088}", "9": "\u{2089}",
        "+": "\u{208A}", "-": "\u{208B}", "=": "\u{208C}", "(": "\u{208D}", ")": "\u{208E}",
        "a": "\u{2090}", "e": "\u{2091}", "i": "\u{1D62}", "j": "\u{2C7C}", "o": "\u{2092}",
        "n": "\u{2099}", "t": "\u{209C}", "x": "\u{2093}",
    ]

    /// Set upright, as LaTeX does.
    private static let functions: Set<String> = [
        "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "max", "min", "lim",
        "det", "dim", "arg", "deg", "gcd", "sup", "inf",
    ]

    private static let symbols: [String: String] = [
        // Greek
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
        "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
        "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
        "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
        "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
        "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
        // Operators and relations
        "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·", "ast": "∗", "star": "⋆",
        "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "leqslant": "≤", "geqslant": "≥",
        "wedge": "∧", "vee": "∨", "cong": "≅", "propto2": "∝", "neq": "≠", "ne": "≠", "approx": "≈",
        "equiv": "≡", "sim": "∼", "simeq": "≃", "propto": "∝", "ll": "≪", "gg": "≫",
        "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇", "in": "∈", "notin": "∉",
        "cup": "∪", "cap": "∩", "setminus": "∖", "emptyset": "∅", "varnothing": "∅",
        // Big operators and calculus
        "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "oint": "∮", "partial": "∂",
        "nabla": "∇", "infty": "∞", "sqrt": "√", "angle": "∠", "perp": "⊥", "parallel": "∥",
        // Arrows
        "rightarrow": "→", "to": "→", "leftarrow": "←", "gets": "←", "leftrightarrow": "↔",
        "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "mapsto": "↦",
        // Dots, spacing and logic
        "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
        "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨", "therefore": "∴",
        "degree": "°", "circ": "∘", "prime": "′", "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ",
        "aleph": "ℵ", "mathbb": "", "displaystyle": "", "limits": "", "nolimits": "",
        "," : "\u{2009}", ";": "\u{2009}", ":": "\u{2009}", "!": "",
    ]
}
