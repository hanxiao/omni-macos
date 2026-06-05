// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Omni",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "OmniKit", targets: ["OmniKit"]),
        .executable(name: "omni-verify", targets: ["omni-verify"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
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
            ]
        ),
        .executableTarget(
            name: "omni-verify",
            dependencies: ["OmniKit"]
        ),
        .testTarget(
            name: "OmniKitTests",
            dependencies: ["OmniKit"],
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
