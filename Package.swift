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
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "OmniKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "omni-verify",
            dependencies: ["OmniKit"]
        ),
        .testTarget(
            name: "OmniKitTests",
            dependencies: ["OmniKit"],
            resources: [.copy("Resources/text_fixtures.json")]
        ),
    ]
)
