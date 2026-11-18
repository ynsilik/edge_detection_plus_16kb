package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

const val TAG: String = "PaperProcessor"

// OpenCV calisma cozunurlugu. Kamera karesi bu genisliğe kucultulur; bulunan
// koseler daha sonra orijinal kare boyutuna geri olceklenir.
private const val WORK_WIDTH = 600.0

/**
 * Kamera onizleme karesinden kart/dokuman dikdortgenini bulur.
 *
 * Eski algoritma tek bir Canny/esik kombinasyonu ve sabit 0.03 approxPolyDP
 * toleransi kullaniyordu; coku kartta 3 nokta uretip "points.size == 4"
 * kontrolune takiliyor ve otomatik tespit hic calismiyordu. Yeni algoritma:
 *  - bilateral filtre + Canny + morfolojik kapatma ile kenarları guclendirir
 *  - dis konturlari alana gore filtreler
 *  - birden fazla approxPolyDP toleransi dener
 *  - alan + dikdortgensellik + merkeze yakinlik ile en iyi adayi secer
 *  - hicbir aday yoksa ekrana 4 kose iceren bir fallback dikdortgen koyar
 *    (kullanicinin manuel olarak kose secmesine gerek kalmaz)
 */
fun processPicture(previewFrame: Mat): Corners? {
    if (previewFrame.empty()) return null

    val originalSize = previewFrame.size()
    val scale = WORK_WIDTH / max(previewFrame.cols().toDouble(), previewFrame.rows().toDouble())
    val working = Mat()
    val gray = Mat()
    val blurred = Mat()
    val edges = Mat()
    val closed = Mat()
    val hierarchy = Mat()

    try {
        Imgproc.resize(previewFrame, working, Size(), scale, scale, Imgproc.INTER_AREA)
        Imgproc.cvtColor(working, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.bilateralFilter(gray, blurred, 9, 40.0, 40.0)
        Imgproc.Canny(blurred, edges, 40.0, 120.0)

        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(5.0, 5.0)
        )
        try {
            Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel)
        } finally {
            kernel.release()
        }

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            closed,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val imageArea = working.rows().toDouble() * working.cols().toDouble()
        var bestQuad: List<Point>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            // Cok kucuk lekeleri ve tam ekran konturu ayikla.
            if (area < imageArea * 0.06 || area > imageArea * 0.97) continue

            val points2f = MatOfPoint2f(*contour.toArray())
            val perimeter = Imgproc.arcLength(points2f, true)
            if (perimeter <= 0.0) continue

            var quad: List<Point>? = null
            for (epsilon in doubleArrayOf(0.015, 0.02, 0.03, 0.04, 0.05, 0.06)) {
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(points2f, approx, epsilon * perimeter, true)
                val candidate = approx.toArray().toList()
                if (candidate.size == 4) {
                    val asMatOfPoint = MatOfPoint(*candidate.toTypedArray())
                    if (Imgproc.isContourConvex(asMatOfPoint)) {
                        quad = candidate
                        break
                    }
                }
            }
            if (quad == null) continue

            val quadMat = MatOfPoint(*quad.toTypedArray())
            val rect = Imgproc.boundingRect(quadMat)
            val rectArea = rect.width.toDouble() * rect.height.toDouble()
            if (rectArea <= 0.0) continue

            val fillRatio = Imgproc.contourArea(quadMat) / rectArea
            if (fillRatio < 0.55) continue

            val aspect = rect.width.toDouble() / rect.height.toDouble()
            if (aspect < 0.35 || aspect > 2.8) continue

            // Merkeze yakin olan kareler tercih edilir (iOS tarafindaki
            // "center-most rect" davranisiyla ayni mantik).
            val dx = (rect.x + rect.width / 2.0 - working.cols() / 2.0) / working.cols()
            val dy = (rect.y + rect.height / 2.0 - working.rows() / 2.0) / working.rows()
            val centerScore = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) * 2.0)

            val score = (area / imageArea) * 100.0 * (0.4 + 0.6 * fillRatio) + centerScore
            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }

        if (bestQuad != null) {
            val ordered = sortQuad(bestQuad)
            val scaled = ordered.map { Point(it.x / scale, it.y / scale) }
            return Corners(scaled, originalSize)
        }

        // Otomatik tespit basarisiz: kullaniciya suruklenebilir 4 kose ver.
        Log.w(TAG, "Otomatik kose tespiti basarisiz, fallback dikdortgen kullaniliyor")
        val w = originalSize.width
        val h = originalSize.height
        val m = 0.08
        return Corners(
            listOf(
                Point(w * m, h * m),
                Point(w * (1 - m), h * m),
                Point(w * (1 - m), h * (1 - m)),
                Point(w * m, h * (1 - m))
            ),
            originalSize
        )
    } catch (e: Exception) {
        Log.e(TAG, "Kose tespiti sirasinda hata", e)
        return null
    } finally {
        hierarchy.release()
        closed.release()
        edges.release()
        blurred.release()
        gray.release()
        working.release()
    }
}

/** 4 noktayi sol-ust, sag-ust, sag-alt, sol-alt sirasina koyar. */
private fun sortQuad(points: List<Point>): List<Point> {
    val cx = points.map { it.x }.average()
    val cy = points.map { it.y }.average()
    val ordered = points.sortedBy { atan2(it.y - cy, it.x - cx) }
    val topLeftIndex = ordered.indices.minByOrNull { ordered[it].x + ordered[it].y } ?: 0
    return (0..3).map { ordered[(topLeftIndex + it) % ordered.size] }
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {

    pts.forEach { Log.i(TAG, "point: $it") }
    val tl = pts[0]
    val tr = pts[1]
    val br = pts[2]
    val bl = pts[3]

    val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
    val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))

    val dw = max(widthA, widthB)
    val maxWidth = java.lang.Double.valueOf(dw).toInt()

    val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
    val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))

    val dh = max(heightA, heightB)
    val maxHeight = java.lang.Double.valueOf(dh).toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)

    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)

    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)

    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)

    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release()
    srcMat.release()
    dstMat.release()
    Log.i(TAG, "crop finish")
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
        srcMat,
        srcMat,
        255.0,
        Imgproc.ADAPTIVE_THRESH_MEAN_C,
        Imgproc.THRESH_BINARY,
        15,
        15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}
