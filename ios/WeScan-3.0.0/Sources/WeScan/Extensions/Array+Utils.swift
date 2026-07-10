//
//  Array+Utils.swift
//  WeScan
//
//  Created by Boris Emorine on 2/8/18.
//  Copyright © 2018 WeTransfer. All rights reserved.
//

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

}
