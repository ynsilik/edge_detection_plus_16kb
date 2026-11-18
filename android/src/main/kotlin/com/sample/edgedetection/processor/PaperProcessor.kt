package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

const val TAG: String = "PaperProcessor"

private const val WORK_WIDTH = 480.0

/**
 * Kamera karesinden kart/dokuman dikdortgenini bulur.
 *
 * Arastirma temelli onemli noktalar:
 *  - Core.normalize(NORM_MINMAX) dusuk varyansli (siyah/tek renk) karelerde
 *    sensor gurultusunu 0-255'e amplifiye eder -> Canny kenar uretir -> morfoloji
 *    bunu butun ekrani kapsayan bir halkaya donusturur. Once stddev kontrolu.
 *  - GaussianBlur BORDER_CONSTANT sinirda yapay gradyan uretir; REFLECT_101 kullan.
 *  - Morfolojiyi 0-padding ile genisletilmis goruntu uzerinde yap, sonra kirp.
 *  - approxPolyDP "tam 4 kose" dayatmesi kareler arasi titreme yapar; bunun
 *    yerine kontur merkezine gore 4 ceyrekte en uzak nokta (jscanify) kullanilir.
 *  - Halka artefaktini RETR_CCOMP hierarchy child kontrolu ile reddet.
 */
fun processPicture(previewFrame: Mat): Corners? {
    if (previewFrame.empty()) return null

    val originalSize = previewFrame.size()
    val scale = WORK_WIDTH / max(previewFrame.cols().toDouble(), previewFrame.rows().toDouble())
    val working = Mat()
    val gray = Mat()
    val blurred = Mat()
    val edges = Mat()
    val padded = Mat()
    val closed = Mat()

    try {
        Imgproc.resize(previewFrame, working, Size(), scale, scale, Imgproc.INTER_AREA)
        Imgproc.cvtColor(working, gray, Imgproc.COLOR_BGR2GRAY)

        // Bos/siyah/duz yuzey kontrolu: normalize etmeden once stddev bak.
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(gray, mean, std)
        val stddev = std.toArray().firstOrNull() ?: 0.0
        mean.release(); std.release()
        if (stddev < 3.0) {
            Log.d(TAG, "Dusuk varyans ($stddev) -> bos kare, tespit yok")
            return null
        }

        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0, Core.BORDER_REFLECT_101)

        val median = medianOf(blurred)
        val lower = max(30.0, min(200.0, 0.66 * median))
        val upper = max(60.0, min(240.0, 1.33 * median))
        Imgproc.Canny(blurred, edges, lower, upper)

        // Morfolojiyi 0-padding ile yap: sinirda ring olusumunu onler.
        val k = 5
        val border = k / 2 + 2
        Imgproc.copyMakeBorder(edges, padded, border, border, border, border, Core.BORDER_CONSTANT, Scalar(0.0))
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(k.toDouble(), k.toDouble()))
        Imgproc.morphologyEx(padded, closed, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val close = Mat(closed, org.opencv.core.Rect(border, border, edges.cols(), edges.rows()))

        val w = close.cols()
        val h = close.rows()
        val band = k / 2 + 4
        clearBorder(close, band)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(close, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE)

        val imageArea = w.toDouble() * h.toDouble()
        val ordered = contours
            .mapIndexed { i, c -> i to c }
            .sortedByDescending { Imgproc.contourArea(it.second) }
            .take(8)

        var bestQuad: List<Point>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for ((index, contour) in ordered) {
            val area = Imgproc.contourArea(contour)
            if (area < imageArea * 0.08 || area > imageArea * 0.85) continue

            // Halka kontrolu: CCOMP'da child varsa ve child alani ana alana yakinssa
            // bu bir kenar halkasidir, gercek kart degil.
            val child = hierarchy.get(0, index).getOrNull(2)?.toInt() ?: -1
            if (child >= 0 && child < contours.size) {
                val childArea = Imgproc.contourArea(contours[child])
                if (childArea > area * 0.80) continue
            }

            val rect = Imgproc.boundingRect(contour)
            if (rect.x <= band || rect.y <= band ||
                rect.x + rect.width >= w - band ||
                rect.y + rect.height >= h - band) continue

            val fill = area / (rect.width.toDouble() * rect.height.toDouble())
            if (fill < 0.70) continue

            val hull = MatOfInt()
            Imgproc.convexHull(contour, hull)
            val hullPoints = MatOfPoint()
            Imgproc.approxPolyDP(contour, hullPoints, 0.02 * Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true), true)
            val hullArea = Imgproc.contourArea(hullPoints)
            hull.release(); hullPoints.release()
            val solidity = if (hullArea > 0) area / hullArea else 0.0
            if (solidity < 0.85) continue

            val quad = cornersFromQuadrants(contour) ?: continue
            if (quad.any { it.x <= band || it.y <= band || it.x >= w - band || it.y >= h - band }) continue

            val top = (distance(quad[0], quad[1]) + distance(quad[3], quad[2])) / 2.0
            val side = (distance(quad[0], quad[3]) + distance(quad[1], quad[2])) / 2.0
            if (top <= 0.0 || side <= 0.0) continue
            val aspect = max(top, side) / min(top, side)
            if (aspect < 1.20 || aspect > 2.10) continue

            val cx = quad.map { it.x }.average()
            val cy = quad.map { it.y }.average()
            val dx = (cx - w / 2.0) / w
            val dy = (cy - h / 2.0) / h
            val centerScore = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) * 2.0)
            val aspectScore = 1.0 - min(1.0, abs(aspect - 1.586) / 0.6)
            val score = (area / imageArea) * 50.0 + centerScore * 20.0 + aspectScore * 20.0 + fill * 10.0

            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }
        hierarchy.release()

        if (bestQuad != null) {
            val scaled = bestQuad!!.map { Point(it.x / scale, it.y / scale) }
            return Corners(scaled, originalSize)
        }
        return null
    } catch (e: Exception) {
        Log.e(TAG, "Kose tespiti hatasi", e)
        return null
    } finally {
        closed.release(); padded.release(); edges.release(); blurred.release(); gray.release(); working.release()
    }
}

private fun cornersFromQuadrants(contour: MatOfPoint): List<Point>? {
    val pts = contour.toArray().toList()
    if (pts.size < 4) return null
    val cx = pts.map { it.x }.average()
    val cy = pts.map { it.y }.average()
    var tl: Point? = null; var tr: Point? = null; var br: Point? = null; var bl: Point? = null
    var dTL = -1.0; var dTR = -1.0; var dBR = -1.0; var dBL = -1.0
    for (p in pts) {
        val dx = p.x - cx; val dy = p.y - cy
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

private fun clearBorder(mat: Mat, band: Int) {
    val w = mat.cols(); val h = mat.rows()
    val b = min(band, min(w, h) / 4)
    for (y in 0 until h) {
        if (y >= b && y < h - b) {
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
    val tl = pts[0]; val tr = pts[1]; val br = pts[2]; val bl = pts[3]
    val dw = max(distance(br, bl), distance(tr, tl))
    val dh = max(distance(tr, br), distance(tl, bl))
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
