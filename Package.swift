// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LIDBench",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "LIDBench",
            path: "Sources/LIDBench"
        ),
    ]
)
