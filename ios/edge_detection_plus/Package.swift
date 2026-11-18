// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "edge_detection_plus",
  platforms: [
    .iOS(.v13)
  ],
  products: [
    .library(name: "edge_detection_plus", targets: ["edge_detection_plus"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework"),
    .package(name: "WeScan", path: "../WeScan-3.0.0")
  ],
  targets: [
    .target(
      name: "edge_detection_plus",
      dependencies: [
        .product(name: "FlutterFramework", package: "FlutterFramework"),
        .product(name: "WeScan", package: "WeScan")
      ],
      path: "../Classes"
    )
  ]
)
