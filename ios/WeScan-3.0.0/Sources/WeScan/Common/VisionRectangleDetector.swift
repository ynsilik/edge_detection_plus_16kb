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
        completion: @escaping ((Quadrilateral?) -> Void)
    ) {
        let rectangleDetectionRequest: VNDetectRectanglesRequest = {
            let rectDetectRequest = VNDetectRectanglesRequest(completionHandler: { request, error in
                guard error == nil, let results = request.results as? [VNRectangleObservation], !results.isEmpty else {
                    completion(nil)
                    return
                }

                let quads: [Quadrilateral] = results.map(Quadrilateral.init)

                guard let bestQuad = quads.best(preferredCenter: preferredCenter) else {
                    completion(nil)
                    return
                }

                let transform = CGAffineTransform.identity
                    .scaledBy(x: width, y: height)

                completion(bestQuad.applying(transform))
            })

            rectDetectRequest.minimumConfidence = 0.6
            rectDetectRequest.maximumObservations = 15
            rectDetectRequest.minimumAspectRatio = 0.2

            return rectDetectRequest
        }()

        do {
            try request.perform([rectangleDetectionRequest])
        } catch {
            completion(nil)
            return
        }

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
