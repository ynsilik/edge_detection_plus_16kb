import Flutter
import PhotosUI
import UIKit
import WeScan

public class SwiftEdgeDetectionPlugin: NSObject, FlutterPlugin {

    /// Aktif tarama oturumu (ayni anda yalnizca bir tane).
    private var session: EdgeScanSession?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "edge_detection", binaryMessenger: registrar.messenger())
        let instance = SwiftEdgeDetectionPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard call.method == "edge_detect" || call.method == "edge_detect_gallery" else {
            result(FlutterMethodNotImplemented)
            return
        }
        guard let args = call.arguments as? [String: Any],
              let saveTo = args["save_to"] as? String, !saveTo.isEmpty else {
            result(FlutterError(code: "invalid_args", message: "save_to is required", details: nil))
            return
        }
        guard session == nil else {
            result(FlutterError(code: "already_active", message: "Edge detection is already active", details: nil))
            return
        }
        guard let presenter = Self.topViewController() else {
            result(FlutterError(code: "no_view_controller", message: "No view controller to present from", details: nil))
            return
        }

        if let v = args["ios_cancel_title"] as? String { WeScanCustomStrings.shared.cancel = v }
        if let v = args["ios_auto_scan_title"] as? String { WeScanCustomStrings.shared.autoScan = v }
        if let v = args["ios_manual_scan_title"] as? String { WeScanCustomStrings.shared.manualScan = v }
        if let v = args["ios_next_title"] as? String { WeScanCustomStrings.shared.next = v }
        if let v = args["ios_edit_title"] as? String { WeScanCustomStrings.shared.editTitle = v }
        if let v = args["ios_review_title"] as? String { WeScanCustomStrings.shared.reviewTitle = v }

        let canUseGallery = args["can_use_gallery"] as? Bool ?? false
        let newSession = EdgeScanSession(saveTo: saveTo) { [weak self] outcome in
            self?.session = nil
            switch outcome {
            case .saved: result(true)
            case .cancelled: result(false)
            case .failed(let message): result(FlutterError(code: "scan_failed", message: message, details: nil))
            }
        }
        session = newSession

        if call.method == "edge_detect" {
            newSession.startCamera(from: presenter, canUseGallery: canUseGallery)
        } else {
            newSession.startGallery(from: presenter)
        }
    }

    /// Sahne tabanli uygulamalarda da calisan en ustteki view controller.
    static func topViewController() -> UIViewController? {
        var window: UIWindow?
        if #available(iOS 13.0, *) {
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let active = scenes.first { $0.activationState == .foregroundActive } ?? scenes.first
            window = active?.windows.first { $0.isKeyWindow } ?? active?.windows.first
        }
        if window == nil {
            window = UIApplication.shared.delegate?.window ?? nil
        }
        var top = window?.rootViewController
        while let presented = top?.presentedViewController, !presented.isBeingDismissed {
            top = presented
        }
        return top
    }
}

/// Bir tarama oturumunu (kamera ya da galeri) bastan sona yonetir ve sonucu
/// tam olarak bir kez bildirir.
final class EdgeScanSession: NSObject {

    enum Outcome {
        case saved
        case cancelled
        case failed(String)
    }

    private let saveTo: String
    private var completion: ((Outcome) -> Void)?
    private weak var presenter: UIViewController?
    private var retainSelf: EdgeScanSession?

    init(saveTo: String, completion: @escaping (Outcome) -> Void) {
        self.saveTo = saveTo
        self.completion = completion
        super.init()
        retainSelf = self // oturum bitene kadar canli kal
    }

    // MARK: - Kamera

    func startCamera(from presenter: UIViewController, canUseGallery: Bool) {
        self.presenter = presenter
        let scanner = makeScanner(image: nil)
        presenter.present(scanner, animated: true) { [weak self, weak scanner] in
            guard let self, let scanner, canUseGallery else { return }
            self.addGalleryButton(to: scanner)
        }
    }

    private func addGalleryButton(to scanner: ImageScannerController) {
        guard let scanView = scanner.viewControllers.first?.view else { return }
        let button = UIButton(type: .system)
        let bundleImage = UIImage(named: "gallery", in: Bundle(for: SwiftEdgeDetectionPlugin.self), compatibleWith: nil)
        var image = bundleImage
        if image == nil, #available(iOS 13.0, *) { image = UIImage(systemName: "photo.on.rectangle") }
        button.setImage(image?.withRenderingMode(.alwaysTemplate), for: .normal)
        button.tintColor = .white
        button.accessibilityLabel = "Gallery"
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: #selector(galleryTappedFromScanner), for: .touchUpInside)
        scanView.addSubview(button)
        NSLayoutConstraint.activate([
            button.widthAnchor.constraint(equalToConstant: 44),
            button.heightAnchor.constraint(equalToConstant: 44),
            button.rightAnchor.constraint(equalTo: scanView.safeAreaLayoutGuide.rightAnchor, constant: -24),
            scanView.safeAreaLayoutGuide.bottomAnchor.constraint(equalTo: button.bottomAnchor, constant: (65.0 / 2) - 10.0)
        ])
    }

    @objc private func galleryTappedFromScanner() {
        guard let presenter else { return finish(.failed("Presenter is gone")) }
        presenter.dismiss(animated: true) { [weak self] in
            self?.startGallery(from: presenter)
        }
    }

    // MARK: - Galeri

    func startGallery(from presenter: UIViewController) {
        self.presenter = presenter
        if #available(iOS 14.0, *) {
            // PHPicker izin istemez; kullanici yalnizca sectigi fotografi paylasir.
            var config = PHPickerConfiguration()
            config.filter = .images
            config.selectionLimit = 1
            let picker = PHPickerViewController(configuration: config)
            picker.delegate = self
            picker.presentationController?.delegate = self
            presenter.present(picker, animated: true)
        } else {
            let picker = UIImagePickerController()
            picker.sourceType = .photoLibrary
            picker.delegate = self
            picker.presentationController?.delegate = self
            presenter.present(picker, animated: true)
        }
    }

    private func presentEditor(with image: UIImage) {
        guard let presenter else { return finish(.failed("Presenter is gone")) }
        let scanner = makeScanner(image: image.normalizedOrientation())
        presenter.present(scanner, animated: true)
    }

    // MARK: - Ortak

    private func makeScanner(image: UIImage?) -> ImageScannerController {
        let scanner = ImageScannerController(image: image, delegate: self)
        scanner.modalPresentationStyle = .fullScreen
        if #available(iOS 13.0, *) {
            scanner.isModalInPresentation = true
            scanner.overrideUserInterfaceStyle = .dark
            scanner.view.backgroundColor = .black
        }
        if #available(iOS 15.0, *) {
            // WeScan #320: iOS 15'te seffaf navigation bar sorunu. Global
            // UIAppearance yerine yalnizca bu navigation bar'a uygulanir.
            let appearance = UINavigationBarAppearance()
            appearance.configureWithOpaqueBackground()
            appearance.titleTextAttributes = [.foregroundColor: UIColor.label]
            appearance.backgroundColor = .systemBackground
            scanner.navigationBar.standardAppearance = appearance
            scanner.navigationBar.scrollEdgeAppearance = appearance
            scanner.navigationBar.compactAppearance = appearance
        }
        return scanner
    }

    private func save(_ image: UIImage) -> Bool {
        // 0.92: gozle ayirt edilemez, dosya ~3-4 kat kucuk (OCR / yukleme icin onemli).
        guard let data = image.jpegData(compressionQuality: 0.92) else { return false }
        let url = URL(fileURLWithPath: saveTo)
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            NSLog("edge_detection: kaydetme hatasi \(error)")
            return false
        }
    }

    private func finish(_ outcome: Outcome) {
        guard let completion else { return }
        self.completion = nil
        DispatchQueue.main.async {
            completion(outcome)
            self.retainSelf = nil
        }
    }
}

// MARK: - ImageScannerControllerDelegate

extension EdgeScanSession: ImageScannerControllerDelegate {
    func imageScannerController(_ scanner: ImageScannerController, didFinishScanningWithResults results: ImageScannerResults) {
        let image = (results.doesUserPreferEnhancedScan ? results.enhancedScan?.image : nil)
            ?? results.croppedScan.image
        let ok = save(image)
        scanner.dismiss(animated: true)
        finish(ok ? .saved : .failed("Could not save scanned image"))
    }

    func imageScannerControllerDidCancel(_ scanner: ImageScannerController) {
        scanner.dismiss(animated: true)
        finish(.cancelled)
    }

    func imageScannerController(_ scanner: ImageScannerController, didFailWithError error: Error) {
        scanner.dismiss(animated: true)
        finish(.failed(error.localizedDescription))
    }
}

// MARK: - Galeri delegeleri

@available(iOS 14.0, *)
extension EdgeScanSession: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        guard let provider = results.first?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else {
            picker.dismiss(animated: true)
            finish(.cancelled)
            return
        }
        provider.loadObject(ofClass: UIImage.self) { [weak self] object, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                picker.dismiss(animated: true) {
                    if let image = object as? UIImage {
                        self.presentEditor(with: image)
                    } else {
                        self.finish(.failed("Could not load selected image"))
                    }
                }
            }
        }
    }
}

extension EdgeScanSession: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        let image = info[.originalImage] as? UIImage
        picker.dismiss(animated: true) { [weak self] in
            guard let self else { return }
            if let image { self.presentEditor(with: image) } else { self.finish(.cancelled) }
        }
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
        finish(.cancelled)
    }
}

extension EdgeScanSession: UIAdaptivePresentationControllerDelegate {
    /// Secici asagi kaydirilarak kapatilirsa.
    func presentationControllerDidDismiss(_ presentationController: UIPresentationController) {
        finish(.cancelled)
    }
}

private extension UIImage {
    /// EXIF yonunu piksellere uygular; WeScan'in kenar tespiti ve kirpmasi dik goruntu bekler.
    func normalizedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
