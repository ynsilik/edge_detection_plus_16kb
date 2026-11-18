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

private const val WORK_WIDTH = 600.0

private data class QuadResult(val quad: List<Point>, val score: Double)

/**
 * Kamera onizleme karesinden kart/dokuman dikdortgenini bulur.
 *
 * iOS Apple Vision kullanir; Android'de OpenCV ile ona yakin bir sonuc icin
 * birden fazla kenar tespiti geçisi (CLAHE + iki Canny esigi + adaptif
 * threshold) denenir ve en iyi aday alan/dikdortgensellik/merkez skoruyla
 * secilir. Hicbir aday yoksa 4 kodeli fallback dikdortgen dondurulur.
 */
fun processPicture(previewFrame: Mat): Corners? {
    if (previewFrame.empty()) return null

    val originalSize = previewFrame.size()
    val scale = WORK_WIDTH / max(previewFrame.cols().toDouble(), previewFrame.rows().toDouble())
    val working = Mat()
    val gray = Mat()
    val equalized = Mat()
    val blurred = Mat()

    try {
        Imgproc.resize(previewFrame, working, Size(), scale, scale, Imgproc.INTER_AREA)
        Imgproc.cvtColor(working, gray, Imgproc.COLOR_BGR2GRAY)

        // CLAHE: degisen isik kosullarinda kontrasti dengele, zayif kenarlari guclendir.
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, equalized)
        Imgproc.bilateralFilter(equalized, blurred, 9, 40.0, 40.0)

        val imageArea = working.rows().toDouble() * working.cols().toDouble()
        var best: QuadResult? = null

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        try {
            // Gecis 1 ve 2: farkli Canny esikleri (dusuk = hassas, yuksek = temiz)
            for ((lo, hi) in arrayOf(Pair(40.0, 120.0), Pair(80.0, 200.0))) {
                val edges = Mat()
                val closed = Mat()
                try {
                    Imgproc.Canny(blurred, edges, lo, hi)
                    Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel)
                    val candidate = bestQuadIn(closed, imageArea, working.cols(), working.rows())
                    if (candidate != null && (best == null || candidate.score > best!!.score)) {
                        best = candidate
                    }
                } finally {
                    edges.release()
                    closed.release()
                }
            }

            // Gecis 3: adaptif esik (Canny'nin kacirdigi dusuk kontrastli kenarlar)
            val binary = Mat()
            try {
                Imgproc.adaptiveThreshold(
                    blurred, binary, 255.0,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV, 21, 10.0
                )
                val candidate = bestQuadIn(binary, imageArea, working.cols(), working.rows())
                if (candidate != null && (best == null || candidate.score > best!!.score)) {
                    best = candidate
                }
            } finally {
                binary.release()
            }
        } finally {
            kernel.release()
        }

        if (best != null) {
            val ordered = sortQuad(best!!.quad)
            val scaled = ordered.map { Point(it.x / scale, it.y / scale) }
            return Corners(scaled, originalSize)
        }

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
        blurred.release()
        equalized.release()
        gray.release()
        working.release()
    }
}

/** Ikili goruntude en iyi 4 kodeli adayi bulur. */
private fun bestQuadIn(binary: Mat, imageArea: Double, imageW: Int, imageH: Int): QuadResult? {
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    try {
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
    } finally {
        hierarchy.release()
    }

    var bestQuad: List<Point>? = null
    var bestScore = Double.NEGATIVE_INFINITY

    for (contour in contours) {
        val area = Imgproc.contourArea(contour)
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

        val dx = (rect.x + rect.width / 2.0 - imageW / 2.0) / imageW
        val dy = (rect.y + rect.height / 2.0 - imageH / 2.0) / imageH
        val centerScore = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) * 2.0)

        val score = (area / imageArea) * 100.0 * (0.4 + 0.6 * fillRatio) + centerScore
        if (score > bestScore) {
            bestScore = score
            bestQuad = quad
        }
    }

    return bestQuad?.let { QuadResult(it, bestScore) }
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
    val tl = pts[0]; val tr = pts[1]; val br = pts[2]; val bl = pts[3]

    val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
    val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))
    val dw = max(widthA, widthB)
    val maxWidth = dw.toInt()

    val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
    val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))
    val dh = max(heightA, heightB)
    val maxHeight = dh.toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)
    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)
    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)
    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)
    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release(); srcMat.release(); dstMat.release()
    Log.i(TAG, "crop finish")
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
        srcMat, srcMat, 255.0,
        Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}
