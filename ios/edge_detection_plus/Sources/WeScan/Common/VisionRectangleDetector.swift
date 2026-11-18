import CoreImage
import Foundation
import Vision

/// Enum encapsulating static functions to detect rectangles from an image.
@available(iOS 11.0, *)
enum VisionRectangleDetector {

    /// Applies preprocessing filters to enhance edge detection.
    private static func preprocessImage(_ ciImage: CIImage) -> CIImage {
        let enhanced = ciImage
            .applyingFilter("CIColorControls", parameters: [
                kCIInputContrastKey: 1.3,
                kCIInputBrightnessKey: 0.05,
                kCIInputSaturationKey: 0.5,
            ])
        return enhanced
    }

    private static func completeImageRequest(
        for request: VNImageRequestHandler,
        width: CGFloat,
        height: CGFloat,
        preferredCenter: CGPoint = CGPoint(x: 0.5, y: 0.5),
        throttleSegmentation: Bool = false,
        completion: @escaping ((Quadrilateral?) -> Void)
    ) {
        let transform = CGAffineTransform.identity.scaledBy(x: width, y: height)

        // 1) Klasik dikdortgen tespiti (hizli, keskin kenarlarda cok isabetli).
        let rectDetectRequest = VNDetectRectanglesRequest()
        rectDetectRequest.minimumConfidence = 0.6
        rectDetectRequest.maximumObservations = 15
        rectDetectRequest.minimumAspectRatio = 0.2

        do {
            try request.perform([rectDetectRequest])
        } catch {
            completion(nil)
            return
        }

        if let results = rectDetectRequest.results, !results.isEmpty,
           let bestQuad = results.map(Quadrilateral.init).best(preferredCenter: preferredCenter) {
            completion(bestQuad.applying(transform))
            return
        }

        // 2) Yedek: iOS 15+ sinir agi tabanli belge segmentasyonu. Dusuk kontrastli
        //    (beyaz kart / acik masa), yuvarlak koseli ya da kismen kapanmis kartlari
        //    klasik dedektorun kacirdigi durumlarda bulur. Canli akista her karede
        //    calistirmamak icin seyreltilir.
        if #available(iOS 15.0, *), !throttleSegmentation || shouldRunSegmentation() {
            let segmentation = VNDetectDocumentSegmentationRequest()
            if (try? request.perform([segmentation])) != nil,
               let results = segmentation.results?.filter({ $0.confidence >= 0.8 }),
               !results.isEmpty,
               let bestQuad = results.map(Quadrilateral.init).best(preferredCenter: preferredCenter),
               isPlausible(bestQuad) {
                completion(bestQuad.applying(transform))
                return
            }
        }

        completion(nil)
    }

    private static var segmentationTick = 0
    private static let segmentationLock = NSLock()

    /// Klasik tespitin bos dondugu her 2 karede bir segmentasyon dener.
    private static func shouldRunSegmentation() -> Bool {
        segmentationLock.lock()
        defer { segmentationLock.unlock() }
        segmentationTick &+= 1
        return segmentationTick % 2 == 0
    }

    /// Segmentasyon cok kucuk / cok ince sonuclari elemek icin basit akil kontrolu
    /// (normalize koordinatlarda).
    private static func isPlausible(_ quad: Quadrilateral) -> Bool {
        let w = Double(hypot(quad.topRight.x - quad.topLeft.x, quad.topRight.y - quad.topLeft.y))
        let h = Double(hypot(quad.bottomLeft.x - quad.topLeft.x, quad.bottomLeft.y - quad.topLeft.y))
        guard w > 0.08, h > 0.08 else { return false }
        let aspect = Swift.max(w, h) / Swift.max(Swift.min(w, h), 0.001)
        return aspect < 4.0

    }

    /// Detects rectangles from the given CVPixelBuffer/CVImageBuffer on iOS 11 and above.
    static func rectangle(forPixelBuffer pixelBuffer: CVPixelBuffer,
                           preferredCenter: CGPoint = CGPoint(x: 0.5, y: 0.5),
                           completion: @escaping ((Quadrilateral?) -> Void)) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let preprocessed = preprocessImage(ciImage)
        let imageRequestHandler = VNImageRequestHandler(ciImage: preprocessed, options: [:])
        VisionRectangleDetector.completeImageRequest(
            for: imageRequestHandler,
            width: CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
            height: CGFloat(CVPixelBufferGetHeight(pixelBuffer)),
            preferredCenter: preferredCenter,
            throttleSegmentation: true,
            completion: completion)
    }

    /// Detects rectangles from the given image on iOS 11 and above.
    static func rectangle(forImage image: CIImage,
                           preferredCenter: CGPoint = CGPoint(x: 0.5, y: 0.5),
                           completion: @escaping ((Quadrilateral?) -> Void)) {
        let preprocessed = preprocessImage(image)
        let imageRequestHandler = VNImageRequestHandler(ciImage: preprocessed, options: [:])
        VisionRectangleDetector.completeImageRequest(
            for: imageRequestHandler, width: image.extent.width,
            height: image.extent.height,
            preferredCenter: preferredCenter,
            completion: completion)
    }

    static func rectangle(
        forImage image: CIImage,
        orientation: CGImagePropertyOrientation,
        preferredCenter: CGPoint = CGPoint(x: 0.5, y: 0.5),
        completion: @escaping ((Quadrilateral?) -> Void)
    ) {
        let preprocessed = preprocessImage(image)
        let orientedImage = preprocessed.oriented(orientation)
        let imageRequestHandler = VNImageRequestHandler(ciImage: orientedImage, orientation: orientation, options: [:])
        VisionRectangleDetector.completeImageRequest(
            for: imageRequestHandler, width: orientedImage.extent.width,
            height: orientedImage.extent.height,
            preferredCenter: preferredCenter,
            completion: completion)
    }

}
