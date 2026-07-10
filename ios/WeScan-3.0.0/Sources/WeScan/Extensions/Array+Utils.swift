import Foundation
import Vision

extension Array where Element == Quadrilateral {

    /// Finds the biggest rectangle within an array of `Quadrilateral` objects.
    func biggest() -> Quadrilateral? {
        let biggestRectangle = self.max(by: { rect1, rect2 -> Bool in
            return rect1.perimeter < rect2.perimeter
        })

        return biggestRectangle
    }

    /// Returns the center point of the quadrilateral in normalized coordinates.
    private func center(of quad: Quadrilateral) -> CGPoint {
        return CGPoint(
            x: (quad.topLeft.x + quad.topRight.x + quad.bottomRight.x + quad.bottomLeft.x) / 4.0,
            y: (quad.topLeft.y + quad.topRight.y + quad.bottomRight.y + quad.bottomLeft.y) / 4.0
        )
    }

    /// Finds the rectangle whose center is closest to the center of the frame (0.5, 0.5 in Vision normalized coordinates).
    func closestToCenter() -> Quadrilateral? {
        let frameCenter = CGPoint(x: 0.5, y: 0.5)
        return self.min(by: { rect1, rect2 -> Bool in
            let center1 = center(of: rect1)
            let center2 = center(of: rect2)
            let dist1 = hypot(center1.x - frameCenter.x, center1.y - frameCenter.y)
            let dist2 = hypot(center2.x - frameCenter.x, center2.y - frameCenter.y)
            return dist1 < dist2
        })
    }

    /// Scores a quadrilateral considering center proximity, area, aspect ratio, and confidence.
    private func scoreQuad(_ quad: Quadrilateral, preferredCenter: CGPoint) -> Double {
        let c = center(of: quad)

        // Center proximity (0-1): closer to preferred center = better
        let maxDist = 0.7
        let dist = hypot(Double(c.x - preferredCenter.x), Double(c.y - preferredCenter.y))
        let centerScore = 1.0 - min(dist / maxDist, 1.0)

        // Area (0-1): bigger rectangle = better
        let w = Double(abs(quad.topRight.x - quad.topLeft.x))
        let h = Double(abs(quad.bottomLeft.y - quad.topLeft.y))
        let area = w * h
        let areaScore = min(area * 3.0, 1.0)

        // Aspect ratio (0-1): prefer ~1.586 (credit card) or ~1.4 (business card)
        let aspect = max(w, h) / max(Swift.min(w, h), 0.001)
        let idealAspect: Double = 1.5
        let aspectScore = 1.0 - min(abs(aspect - idealAspect) / idealAspect, 1.0)

        // Confidence (0-1): from Vision
        let confScore = quad.confidence ?? 0.8

        return centerScore * 0.35 + areaScore * 0.25 + aspectScore * 0.15 + confScore * 0.25
    }

    /// Finds the best quadrilateral using a weighted score of center proximity, area, aspect ratio, and confidence.
    /// - Parameter preferredCenter: The preferred center point (defaults to 0.5, 0.5 for frame center).
    func best(preferredCenter: CGPoint = CGPoint(x: 0.5, y: 0.5)) -> Quadrilateral? {
        return self.max(by: { scoreQuad($0, preferredCenter: preferredCenter) < scoreQuad($1, preferredCenter: preferredCenter) })
    }

}
