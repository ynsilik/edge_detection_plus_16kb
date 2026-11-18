import AVFoundation
import CoreImage
import Foundation

/// Class used to detect rectangles from an image.
enum CIRectangleDetector {

    static let rectangleDetector = CIDetector(ofType: CIDetectorTypeRectangle,
                                              context: CIContext(options: nil),
                                              options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])

    /// Detects rectangles from the given image on iOS 10.
    static func rectangle(forImage image: CIImage, completion: @escaping ((Quadrilateral?) -> Void)) {
        let bestRectangle = rectangle(forImage: image)
        completion(bestRectangle)
    }

    static func rectangle(forImage image: CIImage) -> Quadrilateral? {
        guard let rectangleFeatures = rectangleDetector?.features(in: image) as? [CIRectangleFeature] else {
            return nil
        }

        let quads = rectangleFeatures.map { rectangle in
            return Quadrilateral(rectangleFeature: rectangle)
        }

        return quads.best()
    }
}
