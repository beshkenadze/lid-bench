// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LIDBench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "LIDBench",
            path: "Sources/LIDBench"
        ),
        .executableTarget(
            name: "LIDBenchMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            path: "Sources/LIDBenchMLX"
        ),
    ]
)
