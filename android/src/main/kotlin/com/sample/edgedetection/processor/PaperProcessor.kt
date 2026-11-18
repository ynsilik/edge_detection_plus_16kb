package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

const val TAG: String = "PaperProcessor"

private const val WORK_WIDTH = 600.0
private const val BORDER_BAND = 8.0

/**
 * Kamera karesinden kart/dokuman dikdortgenini bulur.
 *
 * Onemli noktalar (arxiv/search temelli):
 *  - Blur + morpholoji goruntu kenarinda bir "ring" uretir; bu ring en buyuk
 *    kontur olup tum ekrani kapsar. Kenar bandini silip sınıra değen adaylari
 *    reddetmek gerekir (siyah ekranda dev dikdortgen sorunu).
 *  - approxPolyDP ile "tam 4 kose" beklemek kareler arasinda titreme yapar.
 *    Bunun yerine kontur merkezine gore 4 ceyrekte en uzak nokta kullanilir
 *    (jscanify yontemi) - yuvarlak/kopuk koselere de dayaniklidir.
 *  - Canny esikleri goruntunun medyanindan turetilir (sabit esikler isiga
 *    gore fazla hassas/gevsek kalir).
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

    try {
        Imgproc.resize(previewFrame, working, Size(), scale, scale, Imgproc.INTER_AREA)
        Imgproc.cvtColor(working, gray, Imgproc.COLOR_BGR2GRAY)
        Core.normalize(gray, gray, 0.0, 255.0, Core.NORM_MINMAX)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val median = medianOf(blurred)
        val lower = max(30.0, min(200.0, 0.66 * median))
        val upper = max(60.0, min(240.0, 1.33 * median))
        Imgproc.Canny(blurred, edges, lower, upper)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        // Kenar bandini temizle: Canny/morpholoji sinirda ring olusturuyor.
        clearBorder(closed, BORDER_BAND)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()

        val w = working.cols()
        val h = working.rows()
        val imageArea = w.toDouble() * h.toDouble()
        val contoursSorted = contours.sortedByDescending { Imgproc.contourArea(it) }.take(8)

        var bestQuad: List<Point>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (contour in contoursSorted) {
            val area = Imgproc.contourArea(contour)
            if (area < imageArea * 0.05 || area > imageArea * 0.95) continue

            val rect = Imgproc.boundingRect(contour)
            // Sınıra değen adayları reddet (tam ekran ring'i burada elenir).
            if (rect.x < BORDER_BAND || rect.y < BORDER_BAND ||
                rect.x + rect.width > w - BORDER_BAND ||
                rect.y + rect.height > h - BORDER_BAND) continue

            val quad = cornersFromQuadrants(contour) ?: continue
            if (quad.any { it.x < BORDER_BAND || it.y < BORDER_BAND || it.x > w - BORDER_BAND || it.y > h - BORDER_BAND }) continue

            val top = (distance(quad[0], quad[1]) + distance(quad[3], quad[2])) / 2.0
            val side = (distance(quad[0], quad[3]) + distance(quad[1], quad[2])) / 2.0
            if (top <= 0.0 || side <= 0.0) continue
            val aspect = max(top, side) / min(top, side)
            // Kart/genis dikdortgen: ~1.0-2.2 araligi (1.5 ideal).
            if (aspect < 1.05 || aspect > 2.3) continue

            val fillRatio = area / (top * side)
            if (fillRatio < 0.5) continue

            // Merkeze yakinlik tercih edilir.
            val cx = quad.map { it.x }.average()
            val cy = quad.map { it.y }.average()
            val dx = (cx - w / 2.0) / w
            val dy = (cy - h / 2.0) / h
            val centerScore = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) * 2.0)
            val aspectScore = 1.0 - min(1.0, abs(aspect - 1.586) / 1.0)
            val score = (area / imageArea) * 60.0 + centerScore * 20.0 + aspectScore * 20.0 + fillRatio * 10.0

            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }

        if (bestQuad != null) {
            val scaled = bestQuad!!.map { Point(it.x / scale, it.y / scale) }
            return Corners(scaled, originalSize)
        }

        // Tespit yoksa dikdortgen cizme - kullanicinin yanlis yonlendirilmesini
        // engellemek icin bos sonuc dondurulur (onCornersNotDetected cagrilir).
        return null
    } catch (e: Exception) {
        Log.e(TAG, "Kose tespiti hatasi", e)
        return null
    } finally {
        closed.release(); edges.release(); blurred.release(); gray.release(); working.release()
    }
}

/** Kontur merkezine gore 4 ceyrekte en uzak noktalari bulur (TL,TR,BR,BL). */
private fun cornersFromQuadrants(contour: MatOfPoint): List<Point>? {
    val pts = contour.toArray().toList()
    if (pts.size < 4) return null
    val cx = pts.map { it.x }.average()
    val cy = pts.map { it.y }.average()

    var tl: Point? = null; var tr: Point? = null; var br: Point? = null; var bl: Point? = null
    var dTL = -1.0; var dTR = -1.0; var dBR = -1.0; var dBL = -1.0
    for (p in pts) {
        val dx = p.x - cx
        val dy = p.y - cy
        val d = dx * dx + dy * dy
        when {
            dx <= 0 && dy <= 0 -> if (d > dTL) { tl = p; dTL = d }
            dx > 0 && dy <= 0 -> if (d > dTR) { tr = p; dTR = d }
            dx > 0 && dy > 0 -> if (d > dBR) { br = p; dBR = d }
            else -> if (d > dBL) { bl = p; dBL = d }
        }
    }
    return if (tl != null && tr != null && br != null && bl != null) listOf(tl!!, tr!!, br!!, bl!!) else null
}

private fun distance(a: Point, b: Point): Double = sqrt((a.x - b.x).pow(2.0) + (a.y - b.y).pow(2.0))

private fun clearBorder(mat: Mat, band: Double) {
    val w = mat.cols(); val h = mat.rows()
    val b = min(band.toInt(), min(w, h) / 4)
    if (b <= 0) return
    for (y in 0 until h) {
        if (y >= b && y < h - b) {
            // ust/alt sadece kenar satirlari
            for (x in 0 until b) mat.put(y, x, 0.0)
            for (x in (w - b) until w) mat.put(y, x, 0.0)
        } else {
            for (x in 0 until w) mat.put(y, x, 0.0)
        }
    }
}

private fun medianOf(mat: Mat): Double {
    val buf = DoubleArray(mat.rows() * mat.cols())
    mat.get(0, 0, buf)
    buf.sort()
    return buf[buf.size / 2]
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {
    pts.forEach { Log.i(TAG, "point: $it") }
    val tl = pts[0]; val tr = pts[1]; val br = pts[2]; val bl = pts[3]
    val widthA = distance(br, bl); val widthB = distance(tr, tl)
    val dw = max(widthA, widthB)
    val heightA = distance(tr, br); val heightB = distance(tl, bl)
    val dh = max(heightA, heightB)
    val croppedPic = Mat(max(1, dh.toInt()), max(1, dw.toInt()), CvType.CV_8UC4)
    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)
    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)
    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)
    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release(); srcMat.release(); dstMat.release()
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(srcMat, srcMat, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 15.0)
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}
