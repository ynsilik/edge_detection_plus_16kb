// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "edge_detection_plus",
  platforms: [
    .iOS(.v13)
  ],
  products: [
    // Flutter, SPM urun adini eklenti adinin tireli haliyle bekler.
    .library(name: "edge-detection-plus", targets: ["edge_detection_plus"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework")
  ],
  targets: [
    // WeScan kaynaklari paketin icine kopyalandi; dis path bagimliligi yok.
    .target(
      name: "WeScan",
      resources: [.process("Resources")]
    ),
    .target(
      name: "edge_detection_plus",
      dependencies: [
        "WeScan",
        .product(name: "FlutterFramework", package: "FlutterFramework")
      ]
    )
  ]
)
