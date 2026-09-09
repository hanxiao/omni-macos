import Foundation

/// The OCR model is an OPTIONAL add-on: it is never downloaded unless the user asks for it.
///
/// It is a separate ~4 GB artifact from a different model family than the embedding towers, it
/// is not on the indexing path, and most users will never turn it on. So the catalog describes
/// what is available and where it goes; nothing here starts a transfer on its own.
///
/// Weights are hosted as GitHub release assets, which caps a single asset at 2 GiB - hence the
/// sharded layout, and hence a manifest rather than a single URL.
public enum OCRModelCatalog {

    /// The three points on the measured frontier. Every number below is mean over seven complete
    /// pages (each generated to its natural EOS) against the torch bfloat16 reference on M3 Ultra,
    /// one process per build. `CER` is Levenshtein distance over reference length.
    ///
    /// Every variant carries the FastMTP draft head (~70 MB), so decoding speculates by default:
    /// draft 3 tokens, verify them in one target pass, commit what the target agrees with. That
    /// is worth +13-23% and costs nothing in quality.
    ///
    /// The ordering is not the usual one and it is worth stating plainly: 4 bits is NOT free
    /// speed here. Decode on this MoE is bound by fixed per-launch latency at these skinny
    /// shapes rather than by weight bytes, so most of `balanced`'s gain over `fidelity` comes
    /// from 8-bit shared-expert and dense-MLP packs - tensors every token passes through - and
    /// not from any 4-bit tensor. `compact` spends the last 9% of download on 4 bits and pays
    /// for it in accuracy.
    public enum Variant: String, CaseIterable, Sendable {
        /// Routed MoE expert stacks at 8 bits, everything else bf16.
        /// 4.62 GB, 198 tok/s, mean CER 0.0082 on the hard corpus, 8 of 10 pages exact.
        case fidelity
        /// Adds 8-bit shared-expert and dense-MLP packs. Still nothing at 4 bits, and the best
        /// measured quality-per-byte of the three.
        /// 4.53 GB, mean CER 0.0044 on the hard corpus, 9 of 10 pages exact. 236-298 tok/s
        /// depending on page length since the draft chain stopped syncing per token and the
        /// vocabulary shortlist began working; the quoted ~240 is a dense page, not a best case.
        case balanced
        /// Dynamic 4-bit: expert `down` projections and attention at 4 bits (group size 32),
        /// expert `gate_up`, shared expert and dense MLP at 8 bits, router / lm_head / embeddings
        /// / vision left wide. Smallest and fastest, and the accuracy cost is real and lands on
        /// small-print and mixed-script pages rather than on ordinary documents.
        /// 4.13 GB, 203 tok/s, mean CER 0.0465 on the hard corpus, 2 of 10 pages exact - and
        /// handwriting is where it breaks down (CER 0.25 on a clean handwritten page that every
        /// other build transcribes exactly).
        case compact

        public var title: String {
            switch self {
            case .fidelity: return "Highest fidelity"
            case .balanced: return "Balanced"
            case .compact: return "Compact (dynamic 4-bit)"
            }
        }

        /// What the user is actually choosing between, in one line each.
        public var summary: String {
            switch self {
            case .fidelity: return "4.6 GB, ~198 tok/s, CER 0.008"
            case .balanced: return "4.5 GB, ~240 tok/s, CER 0.004"
            case .compact: return "4.1 GB, ~203 tok/s, CER 0.047"
            }
        }

        /// Quantization policy in `Tools/ocr/convert.py` that produces this variant.
        public var policy: String {
            switch self {
            case .fidelity: return "q8"
            case .balanced: return "dyn-k"
            case .compact: return "dyn-j"
            }
        }

        /// Prefix of the release assets for this variant.
        var assetPrefix: String { "jina-ocr-v1-mlx-\(rawValue)-" }
    }

    /// GitHub release the weights are published under. Kept separate from the app's own version
    /// tags so republishing the app does not imply republishing 4 GB of weights.
    public static let releaseTag = "ocr-weights-v1"
    public static let repository = "hanxiao/omni-macos"

    /// Files a variant needs at runtime. `omni-ocr.json` records the quantization policy the
    /// build was made with, which is how a build is identified - never by its directory name.
    public struct Manifest: Sendable {
        public let files: [String]
        public let bytes: Int64
    }

    public static func installDir(for variant: Variant) -> URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        else { return nil }
        return appSupport.appendingPathComponent("Omni/ocr/\(variant.rawValue)")
    }

    /// True when every file the manifest names is present and non-empty.
    ///
    /// A partial download must NOT read as installed: the loader would fail deep inside a
    /// safetensors read with a message about a missing tensor rather than a missing file.
    public static func isInstalled(_ variant: Variant) -> Bool {
        guard let dir = installDir(for: variant),
              let manifest = try? readManifest(at: dir) else { return false }
        let fm = FileManager.default
        let resolved = dir.resolvingSymlinksInPath()
        for file in manifest.files {
            let path = resolved.appendingPathComponent(file).path
            guard let size = try? fm.attributesOfItem(atPath: path)[.size] as? Int64, size > 0 else {
                return false
            }
        }
        return true
    }

    public static func installedVariants() -> [Variant] {
        Variant.allCases.filter(isInstalled)
    }

    /// Read the manifest a completed download left behind.
    /// Bytes an installed build occupies, from its own manifest. Callers size a memory budget
    /// from this: the OCR weights are the single largest thing the app can be asked to hold.
    public static func installedBytes(_ variant: Variant) -> Int {
        guard let dir = installDir(for: variant),
              let manifest = try? readManifest(at: dir) else { return 0 }
        return Int(manifest.bytes)
    }

    static func readManifest(at dir: URL) throws -> Manifest {
        let url = dir.resolvingSymlinksInPath().appendingPathComponent("omni-ocr.json")
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let shards = json["shards"] as? Int else {
            throw OmniError.model("malformed omni-ocr.json in \(dir.path)")
        }
        var files = (1 ... shards).map {
            String(format: "model-%05d-of-%05d.safetensors", $0, shards)
        }
        files += ["tokenizer.json", "tokenizer_config.json", "omni-ocr.json"]
        return Manifest(files: files, bytes: (json["bytes"] as? Int64) ?? 0)
    }

    /// Release asset URL for one file of a variant.
    ///
    /// GitHub release assets are a flat namespace, so the variant is folded into the asset name
    /// and stripped again on the way to disk.
    public static func assetURL(variant: Variant, file: String) -> URL? {
        URL(string: "https://github.com/\(repository)/releases/download/\(releaseTag)/"
            + variant.assetPrefix + file)
    }

    /// The bootstrap problem: the manifest lives INSIDE the download, so the first fetch cannot
    /// read it. `omni-ocr.json` is tiny and is fetched first; every other file is named by it.
    public static let manifestFile = "omni-ocr.json"
}
