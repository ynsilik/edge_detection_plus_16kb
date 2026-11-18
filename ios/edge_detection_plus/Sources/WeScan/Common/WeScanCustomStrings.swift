import Foundation

/// Holds custom string overrides for WeScan UI elements.
/// When set, these values are used instead of NSLocalizedString defaults.
public struct WeScanCustomStrings {
    public var cancel: String?
    public var autoScan: String?
    public var manualScan: String?
    public var next: String?
    public var editTitle: String?
    public var reviewTitle: String?

    public static var shared = WeScanCustomStrings()
}
