// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Omni",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "OmniKit", targets: ["OmniKit"]),
        .library(name: "OmniLink", targets: ["OmniLink"]),
        .executable(name: "omni", targets: ["omni"]),
        .executable(name: "omni-pages", targets: ["omni-pages"]),
        .executable(name: "omni-verify", targets: ["omni-verify"]),
        .executable(name: "ocr-verify", targets: ["ocr-verify"]),
        .executable(name: "profbench", targets: ["profbench"]),
        .executable(name: "filterbench", targets: ["filterbench"]),
        .executable(name: "opentime", targets: ["opentime"]),
        .executable(name: "mutbench", targets: ["mutbench"]),
        .executable(name: "crawlbench", targets: ["crawlbench"]),
        .executable(name: "searchbench", targets: ["searchbench"]),
        .executable(name: "linktool", targets: ["linktool"]),
    ],
    dependencies: [
        // EXACT, not `from:`. 0.32.x makes the OCR model's routed-expert gather ~4 ms a step slower
        // (`ocr-verify x --probe-gather`; CLAUDE.md, "MLX 0.32"), and `from:` would let a resolve
        // update pull it in. Move only after that probe matches on the new version.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.3"),
        // Rust-backed tokenizer (HF `tokenizers` crate), ~6.5x faster than swift-transformers'
        // pure-Swift BPE. Loads the same tokenizer.json, so token ids stay identical (parity).
        .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers", from: "0.5.0"),
    ],
    targets: [
        .target(
            name: "OmniKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            // The Neural Engine programs' graphs and recipes, never their weights (ANEProgram).
            resources: [.copy("Resources/ANE")]
        ),
        // Linked Macs (docs/notes/link.md): identity, pairing, the TLS link and its messages. Apple
        // frameworks only - Network, Security, CryptoKit - and no dependency on the engine, so the
        // protocol is tested on its own.
        .target(
            name: "OmniLink"
        ),
        // The `omni` command line (docs/CLI.md). Foundation only - it is a client of the running
        // app, never a second copy of the engine - and shipped inside the app bundle.
        .executableTarget(
            name: "omni",
            path: "Sources/OmniCLI"
        ),
        // Draws a deck's slides and a workbook's sheets as Quick Look previews them, for their page
        // pictures (DocumentPictures). AppKit and WebKit only; shipped inside the app bundle.
        .executableTarget(
            name: "omni-pages",
            path: "Sources/OmniPages"
        ),
        .executableTarget(
            name: "omni-verify",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "ocr-verify",
            dependencies: [
                "OmniKit",
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "profbench",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "filterbench",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "opentime",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "mutbench",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "crawlbench",
            dependencies: ["OmniKit"]
        ),
        .executableTarget(
            name: "searchbench",
            dependencies: ["OmniKit"]
        ),
        // A linked Mac with no app, for testing a real Omni's answers over the link.
        .executableTarget(
            name: "linktool",
            dependencies: ["OmniLink"]
        ),
        .testTarget(
            name: "OmniLinkTests",
            dependencies: ["OmniLink"]
        ),
        .testTarget(
            name: "OmniCLITests",
            dependencies: ["omni"]
        ),
        .testTarget(
            name: "OmniKitTests",
            dependencies: [
                "OmniKit",
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/text_fixtures.json"),
                .copy("Resources/text_fixtures_nano.json"),
                .copy("Resources/image_ref.safetensors"),
                .copy("Resources/test_image.png"),
                .copy("Resources/video_ref.safetensors"),
                .copy("Resources/audio_ref.safetensors"),
                .copy("Resources/test_audio.wav"),
                .copy("Resources/video_frames"),
            ]
        ),
    ]
)
