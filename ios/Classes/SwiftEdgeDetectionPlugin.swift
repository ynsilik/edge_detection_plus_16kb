import Flutter
import UIKit
import WeScan

public class SwiftEdgeDetectionPlugin: NSObject, FlutterPlugin, UIApplicationDelegate {
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "edge_detection", binaryMessenger: registrar.messenger())
        let instance = SwiftEdgeDetectionPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
        registrar.addApplicationDelegate(instance)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as! Dictionary<String, Any>
        let saveTo = args["save_to"] as! String
        let canUseGallery = args["can_use_gallery"] as? Bool ?? false

        // Set iOS custom titles from Flutter side (ignored on Android)
        if let cancel = args["ios_cancel_title"] as? String { WeScanCustomStrings.shared.cancel = cancel }
        if let auto = args["ios_auto_scan_title"] as? String { WeScanCustomStrings.shared.autoScan = auto }
        if let manual = args["ios_manual_scan_title"] as? String { WeScanCustomStrings.shared.manualScan = manual }
        if let next = args["ios_next_title"] as? String { WeScanCustomStrings.shared.next = next }
        if let edit = args["ios_edit_title"] as? String { WeScanCustomStrings.shared.editTitle = edit }
        if let review = args["ios_review_title"] as? String { WeScanCustomStrings.shared.reviewTitle = review }

        if (call.method == "edge_detect")
        {
            if let viewController = UIApplication.shared.delegate?.window??.rootViewController as? FlutterViewController {
                let destinationViewController = HomeViewController()
                destinationViewController.setParams(saveTo: saveTo, canUseGallery: canUseGallery)
                destinationViewController._result = result
                viewController.present(destinationViewController,animated: true,completion: nil);
            }
        }
        if (call.method == "edge_detect_gallery")
        {
            if let viewController = UIApplication.shared.delegate?.window??.rootViewController as? FlutterViewController {
                let destinationViewController = HomeViewController()
                destinationViewController.setParams(saveTo: saveTo, canUseGallery: canUseGallery)
                destinationViewController._result = result
                destinationViewController.selectPhoto();
            }
        }
    }
}
