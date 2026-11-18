package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

const val TAG: String = "PaperProcessor"

private const val WORK_WIDTH = 600.0

/**
 * Kamera karesinden kart/dokuman dikdortgenini bulur.
 *
 * Onceki surumde koseler "centroid cevre bolgesi" yontemiyle bulunuyordu:
 * kart egildiginde/cevrildiginde bu yontem yanlis koseleri seciyordu.
 * Bu surumde once convexHull + approxPolyDP ile gercek 4 kose aranir,
 * bulunamazsa minAreaRect (dondurulmus dikdortgen) ve en son olarak
 * cevre bolgesi yontemi denenir. Koseler her zaman TL,TR,BR,BL sirasina
 * getirilir.
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
    val dilated = Mat()

    try {
        Imgproc.resize(previewFrame, working, Size(), scale, scale, Imgproc.INTER_AREA)
        Imgproc.cvtColor(working, gray, Imgproc.COLOR_BGR2GRAY)

        val mean = MatOfDouble(); val std = MatOfDouble()
        Core.meanStdDev(gray, mean, std)
        val stddev = std.toArray().firstOrNull() ?: 0.0
        mean.release(); std.release()
        if (stddev < 2.0) {
            Log.d(TAG, "Bos kare (stddev=${stddev.toInt()})")
            return null
        }

        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0, 0.0, Core.BORDER_REFLECT_101)

        val median = medianOf(blurred)
        val lower = (0.55 * median).coerceIn(12.0, 90.0)
        val upper = (1.45 * median).coerceIn(50.0, 200.0)
        Imgproc.Canny(blurred, edges, lower, upper)

        // Kenar cizgisi kopuk olsa bile tek kontur olmasi icin kapat + biraz buyut.
        val k = 9
        val border = k / 2 + 2
        Core.copyMakeBorder(edges, padded, border, border, border, border, Core.BORDER_CONSTANT, Scalar(0.0))
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        Imgproc.morphologyEx(padded, closed, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.dilate(closed, dilated, kernel)
        kernel.release()
        val work = Mat(dilated, Rect(border, border, edges.cols(), edges.rows()))
        val w = work.cols(); val h = work.rows()
        clearBorder(work, 6)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(work, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        Log.d(TAG, "kontur=${contours.size} stddev=${stddev.toInt()} canny=${lower.toInt()}/${upper.toInt()}")

        val imageArea = w.toDouble() * h.toDouble()
        var bestQuad: List<Point>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (contour in contours.sortedByDescending { Imgproc.contourArea(it) }.take(12)) {
            val area = Imgproc.contourArea(contour)
            if (area < imageArea * 0.04 || area > imageArea * 0.97) continue

            // once convex hull: metin/kenar birlesmesinden olusan girintileri temizler.
            val hull = MatOfInt()
            Imgproc.convexHull(contour, hull)
            val hullPts = hull.toArray().map { contour.toArray()[it] }
            hull.release()

            var quad: List<Point>? = null
            var method = "quadrant"

            if (hullPts.size >= 4) {
                val hull2f = MatOfPoint2f(*hullPts.toTypedArray())
                val peri = Imgproc.arcLength(hull2f, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(hull2f, approx, 0.022 * peri, true)
                val approxPts = approx.toArray().toList()
                hull2f.release(); approx.release()
                if (approxPts.size == 4 && Imgproc.isContourConvex(MatOfPoint(*approxPts.toTypedArray()))) {
                    quad = approxPts
                    method = "polyDP"
                }
            }

            if (quad == null) {
                val rr = Imgproc.minAreaRect(MatOfPoint2f(*hullPts.toTypedArray()))
                val box = arrayOfNulls<Point>(4)
                rr.points(box)
                val b = box.filterNotNull()
                if (b.size == 4) { quad = b; method = "minAreaRect" }
            }
            if (quad == null) quad = cornersFromQuadrants(contour)
            if (quad == null || quad.size != 4) continue

            quad = orderPoints(quad)
            if (quad.any { it.x < -8 || it.y < -8 || it.x > w + 8 || it.y > h + 8 }) continue

            val qArea = polygonArea(quad)
            if (qArea < imageArea * 0.04) continue

            val top = (distance(quad[0], quad[1]) + distance(quad[3], quad[2])) / 2.0
            val side = (distance(quad[0], quad[3]) + distance(quad[1], quad[2])) / 2.0
            if (top <= 0.0 || side <= 0.0) continue
            val aspect = max(top, side) / min(top, side)
            if (aspect > 3.2) continue

            val rect = Imgproc.boundingRect(contour)
            val fill = qArea / max(1.0, rect.width.toDouble() * rect.height.toDouble())
            if (fill < 0.45) continue

            val cx = quad.map { it.x }.average()
            val cy = quad.map { it.y }.average()
            val dx = (cx - w / 2.0) / w
            val dy = (cy - h / 2.0) / h
            val centerScore = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) * 2.0)
            val aspectScore = 1.0 - min(1.0, abs(aspect - 1.586) / 1.0)
            val score = (qArea / imageArea) * 55.0 + centerScore * 15.0 + aspectScore * 20.0 + fill * 10.0
            Log.d(TAG, "aday $method area=${(area / imageArea * 100).toInt()}% q=${(qArea / imageArea * 100).toInt()}% aspect=$aspect fill=${fill.toInt()} score=${score.toInt()}")
            if (score > bestScore) { bestScore = score; bestQuad = quad }
        }
        hierarchy.release()

        if (bestQuad != null) {
            Log.d(TAG, "SECILDI score=${bestScore.toInt()}")
            return Corners(bestQuad!!.map { Point(it.x / scale, it.y / scale) }, originalSize)
        }
        Log.d(TAG, "Kose bulunamadi")
        return null
    } catch (e: Exception) {
        Log.e(TAG, "Kose tespiti hatasi", e)
        return null
    } finally {
        dilated.release(); closed.release(); padded.release(); edges.release(); blurred.release(); gray.release(); working.release()
    }
}

/** 4 noktayi sira ile TL, TR, BR, BL olarak duzenler. */
private fun orderPoints(pts: List<Point>): List<Point> {
    val a = pts.take(4)
    val sum = a.map { it.x + it.y }
    val diff = a.map { it.x - it.y }
    return listOf(
        a[sum.indexOf(sum.minOrNull()!!)],
        a[diff.indexOf(diff.maxOrNull()!!)],
        a[sum.indexOf(sum.maxOrNull()!!)],
        a[diff.indexOf(diff.minOrNull()!!)]
    )
}

private fun polygonArea(pts: List<Point>): Double {
    var area = 0.0
    for (i in pts.indices) {
        val p1 = pts[i]; val p2 = pts[(i + 1) % pts.size]
        area += p1.x * p2.y - p2.x * p1.y
    }
    return abs(area) / 2.0
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
    for (y in 0 until h) {
        if (y >= band && y < h - band) {
            for (x in 0 until band) mat.put(y, x, 0.0)
            for (x in (w - band) until w) mat.put(y, x, 0.0)
        } else {
            for (x in 0 until w) mat.put(y, x, 0.0)
        }
    }
}

private fun medianOf(mat: Mat): Double {
    // blurred Mat CV_8UC1 oldugu icin DoubleArray ile okunamaz (exception).
    val values = ByteArray(mat.rows() * mat.cols())
    var i = 0
    for (y in 0 until mat.rows()) {
        val row = ByteArray(mat.cols())
        mat.get(y, 0, row)
        for (x in row.indices) { values[i++] = row[x] }
    }
    val sorted = values.map { it.toInt() and 0xFF }.sorted()
    return sorted[sorted.size / 2].toDouble()
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
