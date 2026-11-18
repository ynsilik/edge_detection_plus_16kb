package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

const val TAG: String = "PaperProcessor"

/** Canli onizleme icin calisma cozunurlugu (uzun kenar, px). */
const val PREVIEW_WORK_SIZE = 640.0

/** Cekilen fotograf / galeride kose iyilestirmesinin calistigi cozunurluk. */
const val STILL_REFINE_SIZE = 1600.0

/**
 * Bir karede kart/dokuman dortgenini bulur.
 *
 * Yaklasim (tek esikli Canny + minAreaRect yerine):
 *  1. Gri ve doygunluk (saturation) kanallari ayri ayri islenir. Beyaz kart / acik
 *     masa gibi dusuk parlaklik kontrasti olan sahnelerde renk kanali kurtarir.
 *  2. Her kanal icin 3 ikili harita uretilir: yarim cozunurlukte (doku bastirilmis)
 *     otomatik ve dusuk esikli Canny, ve asindirilmis Otsu lekesi.
 *  3. Her konturun convex hull'u dortgene indirgenir; ardindan konturun kendi
 *     noktalari her kenara atanip dogru uydurulur (yuvarlak koseler, parmakla
 *     kapanmis koseler icin dayanikli).
 *  4. Her aday; kenar destegi, aci duzgunlugu ve alan ile puanlanir. Kenar destegi
 *     olmayan "hayalet" dortgenler ve bir kenari kare sinirina yapisik adaylar
 *     (tamami gorunmeyen nesneler) elenir.
 *  5. [hint] verilirse (onceki karenin dortgeni) ona yakin adaylar hafifce
 *     odullendirilir (yalnizca esitlik bozucu; yanlis adaya kilitlenmeyi onler).
 *  6. Kazanan dortgenin her kenari, kenara dik en guclu ve ayni yonlu parlaklik
 *     gecisine robust dogru uydurmasiyla iyilestirilir (yazi / golge cizgilerine
 *     yapismaz); koseler bu dogrularin kesisimidir.
 *
 * Donen koseler her zaman TL, TR, BR, BL sirasinda ve [frame] koordinatlarindadir.
 */
fun processPicture(
    frame: Mat,
    hint: List<Point>? = null,
    workSize: Double = PREVIEW_WORK_SIZE,
    hintTolerance: Double = 0.05
): Corners? {
    if (frame.empty()) return null
    val scale = min(1.0, workSize / max(frame.cols(), frame.rows()).toDouble())

    val small = Mat()
    val gray = Mat()
    val sat = Mat()
    val support = Mat()
    val rawEdges = Mat()
    try {
        if (scale < 1.0) {
            Imgproc.resize(frame, small, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            frame.copyTo(small)
        }
        val w = small.cols()
        val h = small.rows()
        // Tam calisma cozunurlugunde yalnizca gri (kose iyilestirme icin).
        toGray(small, gray)

        // Aday uretimi YARIM cozunurlukte: ~4 kat hizli ve doku daha iyi bastirilir.
        // Renk donusumu (HSV) de yarim cozunurlukte yapilir.
        val hw = w / 2
        val hh = h / 2
        val halfColor = Mat()
        Imgproc.resize(small, halfColor, Size(hw.toDouble(), hh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val halfGray = Mat()
        splitGrayAndSaturation(halfColor, halfGray, sat)
        halfColor.release()
        if (stdDev(halfGray) < 4.0) { // bos / kapali kamera karesi
            halfGray.release()
            return null
        }
        val channels = ArrayList<Mat>(2)
        channels.add(halfGray)
        if (!sat.empty() && stdDev(sat) >= 8.0) channels.add(sat)

        val maps = ArrayList<Mat>()
        rawEdges.create(hh, hw, CvType.CV_8UC1)
        rawEdges.setTo(org.opencv.core.Scalar(0.0))
        for (ch in channels) buildBinaryMaps(ch, maps, rawEdges)
        halfGray.release()

        // Kenar destegi haritasi (asil tolerans sideSupport'taki dik aramada).
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.dilate(rawEdges, support, k3)
        k3.release()
        val supportBytes = Buffers.get().bytes(hw * hh)
        support.get(0, 0, supportBytes)

        val halfHint = hint?.takeIf { it.size == 4 }?.map { Point(it.x * scale / 2, it.y * scale / 2) }

        var best: List<Point>? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (map in maps) {
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(map, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE)
            hierarchy.release()
            for (c in contours) {
                val quad = quadFromContour(c, hw, hh)
                c.release()
                if (quad == null) continue
                val score = scoreQuad(quad, hw, hh, supportBytes, halfHint, hintTolerance) ?: continue
                if (score > bestScore) {
                    bestScore = score
                    best = quad.map { Point(it.x * 2, it.y * 2) }
                }
            }
            map.release()
        }

        val result = refineQuad(best ?: return null, gray, w, h)
        return Corners(result.map { Point(it.x / scale, it.y / scale) }, frame.size())
    } catch (e: Exception) {
        Log.e(TAG, "Kose tespiti hatasi", e)
        return null
    } finally {
        small.release(); gray.release(); sat.release(); support.release(); rawEdges.release()
    }
}

/**
 * Verilen dortgenin koselerini [frame] uzerinde, uzun kenari en fazla [workSize] px
 * olacak cozunurlukte gradyan tabanli olarak iyilestirir. Tespit dusuk cozunurlukte
 * (kararli), hassasiyet yuksek cozunurlukte elde edilir.
 */
fun refineCorners(frame: Mat, quad: List<Point>, workSize: Double = STILL_REFINE_SIZE): List<Point> {
    if (frame.empty() || quad.size != 4) return quad
    val scale = min(1.0, workSize / max(frame.cols(), frame.rows()).toDouble())
    val small = Mat()
    val gray = Mat()
    val sat = Mat()
    return try {
        if (scale < 1.0) Imgproc.resize(frame, small, Size(), scale, scale, Imgproc.INTER_AREA)
        else frame.copyTo(small)
        toGray(small, gray)
        val q = orderQuad(quad).map { Point(it.x * scale, it.y * scale) }
        refineQuad(q, gray, gray.cols(), gray.rows()).map { Point(it.x / scale, it.y / scale) }
    } catch (e: Exception) {
        Log.e(TAG, "Kose iyilestirme hatasi", e)
        quad
    } finally {
        small.release(); gray.release(); sat.release()
    }
}

/** Iki dortgenin ortalama kose uzakligi / [diag]. */
fun meanCornerDistance(a: List<Point>, b: List<Point>, diag: Double): Double =
    (0 until 4).sumOf { distance(a[it], b[it]) } / 4.0 / diag

// -------------------------------------------------------------------------
// On isleme
// -------------------------------------------------------------------------

private fun toGray(src: Mat, gray: Mat) {
    when (src.channels()) {
        1 -> src.copyTo(gray)
        3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)
        else -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
    }
}

private fun splitGrayAndSaturation(src: Mat, gray: Mat, sat: Mat) {
    when (src.channels()) {
        1 -> src.copyTo(gray)
        3 -> {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)
            val hsv = Mat()
            Imgproc.cvtColor(src, hsv, Imgproc.COLOR_RGB2HSV)
            Core.extractChannel(hsv, sat, 1)
            hsv.release()
        }
        else -> {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            val rgb = Mat()
            val hsv = Mat()
            Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
            Core.extractChannel(hsv, sat, 1)
            rgb.release(); hsv.release()
        }
    }
}

/**
 * Kanal basina 3 ikili harita uretir; ham kenarlari [rawEdges]'e ekler.
 *
 * [channel] zaten YARIM cozunurluktedir; Canny medyan bulanikliktan sonra calisir: deri, ahsap,
 * kumas gibi dokular tamamen kaybolur, kart siniri ise temiz ve kapali kalir.
 * (Tam cozunurlukte Canny dokuyu kart kenarina yapistirip konturu bozuyordu.)
 * Otsu lekesi asindirilir; kartin arka plandaki baska bir acik bolgeye ince bir
 * seritle degmesi lekeyi birlestirmesin.
 */
private fun buildBinaryMaps(channel: Mat, out: MutableList<Mat>, rawEdges: Mat) {
    val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))

    val md = Mat()
    Imgproc.medianBlur(channel, md, 5)
    val median = medianOf(md)
    val lo = (0.67 * median).coerceIn(10.0, 120.0)
    val hi = (1.33 * median).coerceIn(lo * 2.0, 250.0)
    for ((a, b) in listOf(lo to hi, 12.0 to 36.0)) {
        val edges = Mat()
        Imgproc.Canny(md, edges, a, b)
        Core.bitwise_or(rawEdges, edges, rawEdges)
        Imgproc.dilate(edges, edges, k3)
        out.add(edges)
    }
    md.release()

    val blurred = Mat()
    Imgproc.GaussianBlur(channel, blurred, Size(3.0, 3.0), 0.0)
    val otsu = Mat()
    Imgproc.threshold(blurred, otsu, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
    val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
    Imgproc.erode(otsu, otsu, k5)
    k5.release()
    out.add(otsu)

    k3.release()
    blurred.release()
}

// -------------------------------------------------------------------------
// Kontur -> dortgen
// -------------------------------------------------------------------------

private fun quadFromContour(contour: MatOfPoint, w: Int, h: Int): List<Point>? {
    val bound = Imgproc.boundingRect(contour)
    if (bound.width < w * 0.15 || bound.height < h * 0.15) return null

    val hullIdx = MatOfInt()
    Imgproc.convexHull(contour, hullIdx)
    val cPts = contour.toArray()
    val hullPts = hullIdx.toArray().map { cPts[it] }
    hullIdx.release()
    if (hullPts.size < 4) return null

    val hull2f = MatOfPoint2f(*hullPts.map { Point(it.x, it.y) }.toTypedArray())
    val imageArea = w.toDouble() * h
    val hullArea = Imgproc.contourArea(hull2f)
    if (hullArea < imageArea * 0.05 || hullArea > imageArea * 0.985) {
        hull2f.release(); return null
    }
    val peri = Imgproc.arcLength(hull2f, true)

    var quad: List<Point>? = null
    val approx = MatOfPoint2f()
    for (eps in doubleArrayOf(0.01, 0.015, 0.02, 0.03, 0.045, 0.06)) {
        Imgproc.approxPolyDP(hull2f, approx, eps * peri, true)
        val pts = approx.toArray().toList()
        if (pts.size == 4) { quad = orderQuad(pts); break }
        if (pts.size in 5..8) {
            // Yuvarlak / kapanmis koseler: en uzun 4 kenarin dogrularini kesistir.
            val fitted = quadFromLongestSides(pts)
            if (fitted != null) {
                val a = polygonArea(fitted)
                if (a >= hullArea * 0.95 && a <= hullArea * 1.2) { quad = orderQuad(fitted); break }
            }
        }
        if (pts.size < 4) break
    }
    approx.release(); hull2f.release()
    return quad?.let { fitSides(it, cPts) }
}

/**
 * Kontur noktalarini kaba dortgenin kenarlarina atar (kose bolgeleri haric) ve her
 * kenara robust dogru uydurur. Yuvarlak koseli kartlarda approxPolyDP'nin egik /
 * iceride kalan kenarlarini duzeltir. Sonuc makul degilse [q] doner.
 */
private fun fitSides(q: List<Point>, pts: Array<Point>): List<Point> {
    if (pts.size < 16) return q
    var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
    var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE
    for (p in pts) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y
    }
    val diag = hypot(maxX - minX, maxY - minY)
    val lines = ArrayList<DoubleArray>(4)
    for (i in 0 until 4) {
        val a = q[i]
        val b = q[(i + 1) % 4]
        val len = distance(a, b)
        if (len < 1.0) return q
        val ux = (b.x - a.x) / len
        val uy = (b.y - a.y) / len
        val sel = ArrayList<Point>()
        for (p in pts) {
            val rx = p.x - a.x
            val ry = p.y - a.y
            val t = (rx * ux + ry * uy) / len
            val off = abs(-rx * uy + ry * ux)
            if (t > 0.12 && t < 0.88 && off < 0.04 * diag) sel.add(p)
        }
        if (sel.size < 8) {
            lines.add(doubleArrayOf(ux, uy, a.x, a.y))
            continue
        }
        val m = MatOfPoint2f(*sel.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(m, line, Imgproc.DIST_HUBER, 0.0, 0.01, 0.01)
        val v = DoubleArray(4)
        for (r in 0 until 4) v[r] = line.get(r, 0)[0]
        m.release(); line.release()
        lines.add(v)
    }
    val out = ArrayList<Point>(4)
    for (i in 0 until 4) {
        val l1 = lines[(i + 3) % 4]
        val l2 = lines[i]
        out.add(
            lineIntersection(
                Point(l1[2], l1[3]), Point(l1[2] + l1[0], l1[3] + l1[1]),
                Point(l2[2], l2[3]), Point(l2[2] + l2[0], l2[3] + l2[1])
            ) ?: return q
        )
    }
    if (!isConvex(out) || (0 until 4).any { distance(out[it], q[it]) > 0.1 * diag }) return q
    return out
}

/** Cokgenin en uzun 4 kenarini (sirayi koruyarak) secip ardisik dogrulari kesistirir. */
private fun quadFromLongestSides(poly: List<Point>): List<Point>? {
    val n = poly.size
    val sides = (0 until n).map { i -> i to distance(poly[i], poly[(i + 1) % n]) }
    val chosen = sides.sortedByDescending { it.second }.take(4).map { it.first }.sorted()
    val out = ArrayList<Point>(4)
    for (k in 0 until 4) {
        val i = chosen[k]
        val j = chosen[(k + 1) % 4]
        val p = lineIntersection(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]) ?: return null
        out.add(p)
    }
    return out
}

private fun lineIntersection(a1: Point, a2: Point, b1: Point, b2: Point): Point? {
    val d = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x)
    if (abs(d) < 1e-6) return null
    val t1 = a1.x * a2.y - a1.y * a2.x
    val t2 = b1.x * b2.y - b1.y * b2.x
    return Point(
        (t1 * (b1.x - b2.x) - (a1.x - a2.x) * t2) / d,
        (t1 * (b1.y - b2.y) - (a1.y - a2.y) * t2) / d
    )
}

// -------------------------------------------------------------------------
// Dogrulama + puanlama
// -------------------------------------------------------------------------

private fun scoreQuad(
    q: List<Point>,
    w: Int,
    h: Int,
    support: ByteArray,
    hint: List<Point>?,
    hintTolerance: Double
): Double? {
    val tol = 0.04 * max(w, h)
    if (q.any { it.x < -tol || it.y < -tol || it.x > w + tol || it.y > h + tol }) return null
    if (!isConvex(q)) return null

    // Acilar 60..120 derece arasinda olmali (perspektif payi ile).
    var maxCos = 0.0
    for (i in 0 until 4) {
        maxCos = max(maxCos, abs(angleCos(q[(i + 3) % 4], q[i], q[(i + 1) % 4])))
    }
    if (maxCos > 0.5) return null

    val sides = (0 until 4).map { distance(q[it], q[(it + 1) % 4]) }
    if (sides.minOrNull()!! < 0.12 * min(w, h)) return null
    val horiz = (sides[0] + sides[2]) / 2.0
    val vert = (sides[1] + sides[3]) / 2.0
    if (max(horiz, vert) / min(horiz, vert) > 4.0) return null

    // Bir kenari kare sinirina yapisik aday: nesnenin tamami gorunmuyor (or. klavye
    // tuslari, masa kenari) ya da goruntu cercevesinin kendisi -> ret.
    val m = max(3.0, 0.015 * max(w, h))
    for (i in 0 until 4) {
        val a = q[i]
        val b = q[(i + 1) % 4]
        if ((a.x <= m && b.x <= m) || (a.y <= m && b.y <= m) ||
            (a.x >= w - 1 - m && b.x >= w - 1 - m) || (a.y >= h - 1 - m && b.y >= h - 1 - m)
        ) return null
    }

    // Kenar destegi: her kenar gercek bir kenar cizgisi uzerinde mi?
    var supportSum = 0.0
    for (i in 0 until 4) {
        val s = sideSupport(q[i], q[(i + 1) % 4], w, h, support)
        if (s < 0.45) return null
        supportSum += s
    }
    val meanSupport = supportSum / 4.0
    if (meanSupport < 0.6) return null

    val areaFrac = polygonArea(q) / (w.toDouble() * h)
    var score = areaFrac * 3.0 + meanSupport + (1.0 - maxCos) * 0.3

    if (hint != null) {
        // Yalnizca esitlik bozucu: gercekten daha iyi bir aday onceki secimi gecebilmeli.
        val diag = hypot(w.toDouble(), h.toDouble())
        val d = (0 until 4).sumOf { distance(q[it], hint[it]) } / 4.0 / diag
        score += 0.3 * max(0.0, 1.0 - d / hintTolerance)
    }
    return score
}

/**
 * Kenar boyunca orneklenen her noktada, kenara dik +-%1'lik bir bantta kenar pikseli
 * var mi? Isabet orani dondurulur. Kaba dortgen birkac piksel kayik olabilir;
 * asil hassasiyeti [refineQuad] saglar.
 */
private fun sideSupport(a: Point, b: Point, w: Int, h: Int, support: ByteArray): Double {
    val len = distance(a, b)
    if (len < 1.0) return 0.0
    val samples = max(16, (len / 3.0).toInt())
    val r = max(3, (max(w, h) * 0.01).toInt())
    val nx = -(b.y - a.y) / len
    val ny = (b.x - a.x) / len
    var hit = 0
    var total = 0
    // Kose civarini (%8) atla: parmak / yuvarlak kose kenar destegini dusurmesin.
    for (i in 0..samples) {
        val t = 0.08 + 0.84 * i / samples
        val px = a.x + (b.x - a.x) * t
        val py = a.y + (b.y - a.y) * t
        if (px < 0 || py < 0 || px >= w || py >= h) continue
        total++
        for (o in -r..r) {
            val x = (px + nx * o).toInt()
            val y = (py + ny * o).toInt()
            if (x in 0 until w && y in 0 until h && support[y * w + x].toInt() != 0) {
                hit++
                break
            }
        }
    }
    return if (total < samples / 2) 0.0 else hit.toDouble() / total
}

// -------------------------------------------------------------------------
// Kose iyilestirme
// -------------------------------------------------------------------------

/** Koseleri gradyan tabanli iki turlu kenar uydurmasiyla iyilestirir; sonuc makul degilse [q] doner. */
private fun refineQuad(q: List<Point>, gray: Mat, w: Int, h: Int): List<Point> {
    val coarseR = max(4, (max(w, h) / 80.0).toInt() + 3)
    // Gradyan yalnizca dortgenin cevresinde (ROI) hesaplanir: kart kucukse cok daha ucuz.
    val pad = coarseR + 6
    val x0 = max(0, (q.minOf { it.x } - pad).toInt())
    val y0 = max(0, (q.minOf { it.y } - pad).toInt())
    val x1 = min(w, (q.maxOf { it.x } + pad).toInt() + 1)
    val y1 = min(h, (q.maxOf { it.y } + pad).toInt() + 1)
    val rw = x1 - x0
    val rh = y1 - y0
    if (rw < 8 || rh < 8) return q

    val g = Mat(); val gx = Mat(); val gy = Mat()
    val roi = gray.submat(y0, y1, x0, x1)
    val buf = Buffers.get()
    val ax = buf.gx(rw * rh)
    val ay = buf.gy(rw * rh)
    try {
        Imgproc.GaussianBlur(roi, g, Size(3.0, 3.0), 0.0)
        Imgproc.Sobel(g, gx, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(g, gy, CvType.CV_16S, 0, 1, 3)
        gx.get(0, 0, ax)
        gy.get(0, 0, ay)
    } finally {
        g.release(); gx.release(); gy.release(); roi.release()
    }
    val local = q.map { Point(it.x - x0, it.y - y0) }
    // 1. tur genis arama, 2. tur bulunan dogru etrafinda dar arama.
    val first = refinePass(local, ax, ay, rw, rh, coarseR) ?: return q
    val second = (refinePass(first, ax, ay, rw, rh, 3) ?: first).map { Point(it.x + x0, it.y + y0) }

    // Akil kontrolu: koseler cok kaymamali, dortgen gecerli kalmali.
    val maxShift = 0.05 * hypot(w.toDouble(), h.toDouble())
    if ((0 until 4).any { distance(second[it], q[it]) > maxShift }) return q
    if (!isConvex(second)) return q
    val a0 = polygonArea(q)
    val a1 = polygonArea(second)
    if (a1 < a0 * 0.85 || a1 > a0 * 1.15) return q
    return second
}

/**
 * Her kenar boyunca orneklenen noktalarda, kenara dik dogrultuda (Sobel gradyaninin
 * normal uzerindeki izdusumu) DISARIDAN ICERIYE taranir ve o dogrultudaki en guclu
 * gecisin en az yarisi kadar guclu olan ILK yerel tepe secilir:
 *  - Kart siniri her zaman kartin uzerindeki yazilardan daha disaridadir; yazi
 *    gecisleri daha guclu olsa bile secilmez.
 *  - Kartin altindaki yumusak golgenin zemine gecisi zayif kalir, esigi gecemez.
 * Sinir kenar boyunca ayni yonlu bir gecis oldugundan cogunluk yonune uymayan
 * noktalar atilir, kalanlara Huber dogru uydurulur. Koseler dogrularin kesisimidir.
 */
private fun refinePass(
    q: List<Point>, ax: ShortArray, ay: ShortArray, w: Int, h: Int, r: Int
): List<Point>? {
    val lines = ArrayList<DoubleArray>(4) // [vx, vy, x0, y0]
    val cx = q.sumOf { it.x } / 4.0
    val cy = q.sumOf { it.y } / 4.0
    val span = 2 * r + 1
    val prof = DoubleArray(span)
    for (i in 0 until 4) {
        val a = q[i]
        val b = q[(i + 1) % 4]
        val len = distance(a, b)
        if (len < 1.0) return null
        val dx = (b.x - a.x) / len
        val dy = (b.y - a.y) / len
        // Normal her zaman disari (merkezden uzaga) baksin.
        var nx = -dy
        var ny = dx
        if (nx * ((a.x + b.x) / 2 - cx) + ny * ((a.y + b.y) / 2 - cy) < 0) { nx = -nx; ny = -ny }
        val samples = max(24, (len / 3).toInt())
        val xs = DoubleArray(samples + 1)
        val ys = DoubleArray(samples + 1)
        val sg = IntArray(samples + 1)
        var n = 0
        for (k in 0..samples) {
            val t = 0.08 + 0.84 * k / samples
            val px = a.x + (b.x - a.x) * t
            val py = a.y + (b.y - a.y) * t
            var vmax = 0.0
            for (j in 0 until span) {
                val off = r - j // disaridan iceriye
                val x = (px + nx * off + 0.5).toInt()
                val y = (py + ny * off + 0.5).toInt()
                if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { prof[j] = 0.0; continue }
                val idx = y * w + x
                prof[j] = ax[idx] * nx + ay[idx] * ny
                if (abs(prof[j]) > vmax) vmax = abs(prof[j])
            }
            if (vmax < 60.0) continue // zayif gecis: kanit sayma
            val thresh = vmax * 0.5
            var pick = -1
            for (j in 0 until span) {
                val m = abs(prof[j])
                if (m < thresh) continue
                val prev = if (j > 0) abs(prof[j - 1]) else 0.0
                val next = if (j < span - 1) abs(prof[j + 1]) else 0.0
                if (m >= prev && m >= next) { pick = j; break }
            }
            if (pick < 0) continue
            val bestOff = r - pick
            val bestV = prof[pick]
            xs[n] = px + nx * bestOff
            ys[n] = py + ny * bestOff
            sg[n] = if (bestV > 0) 1 else -1
            n++
        }
        val pos = (0 until n).count { sg[it] > 0 }
        val major = if (pos * 2 >= n) 1 else -1
        val pts = (0 until n).filter { sg[it] == major }.map { Point(xs[it], ys[it]) }
        if (pts.size < samples * 0.35) {
            lines.add(doubleArrayOf(dx, dy, a.x, a.y)) // yeterli kanit yok: kenari koru
            continue
        }
        val mat = MatOfPoint2f(*pts.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(mat, line, Imgproc.DIST_HUBER, 0.0, 0.01, 0.01)
        val v = DoubleArray(4)
        for (row in 0 until 4) v[row] = line.get(row, 0)[0]
        mat.release(); line.release()
        lines.add(v)
    }
    val out = ArrayList<Point>(4)
    for (i in 0 until 4) {
        val l1 = lines[(i + 3) % 4] // kosenin onceki kenari
        val l2 = lines[i]
        out.add(
            lineIntersection(
                Point(l1[2], l1[3]), Point(l1[2] + l1[0], l1[3] + l1[1]),
                Point(l2[2], l2[3]), Point(l2[2] + l2[0], l2[3] + l2[1])
            ) ?: return null
        )
    }
    return out
}

// -------------------------------------------------------------------------
// Tekrar kullanilan tamponlar (her karede ~1.5MB ayirip GC takilmasina yol acmamak icin)
// -------------------------------------------------------------------------

private class Buffers {
    private var b = ByteArray(0)
    private var x = ShortArray(0)
    private var y = ShortArray(0)
    fun bytes(n: Int): ByteArray { if (b.size != n) b = ByteArray(n); return b }
    fun gx(n: Int): ShortArray { if (x.size != n) x = ShortArray(n); return x }
    fun gy(n: Int): ShortArray { if (y.size != n) y = ShortArray(n); return y }

    companion object {
        private val local = object : ThreadLocal<Buffers>() {
            override fun initialValue() = Buffers()
        }
        fun get(): Buffers = local.get()!!
    }
}

// -------------------------------------------------------------------------
// Geometri yardimcilari
// -------------------------------------------------------------------------

/** Koseleri TL, TR, BR, BL sirasina dizer (donmus kartlarda da kararli). */
fun orderQuad(pts: List<Point>): List<Point> {
    val cx = pts.sumOf { it.x } / pts.size
    val cy = pts.sumOf { it.y } / pts.size
    val sorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) } // ekranda saat yonu
    val start = sorted.indices.minByOrNull { sorted[it].x + sorted[it].y } ?: 0
    return List(4) { Point(sorted[(start + it) % 4].x, sorted[(start + it) % 4].y) }
}

private fun isConvex(q: List<Point>): Boolean {
    var sign = 0
    for (i in 0 until 4) {
        val a = q[i]; val b = q[(i + 1) % 4]; val c = q[(i + 2) % 4]
        val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
        val s = if (cross > 0) 1 else if (cross < 0) -1 else 0
        if (s == 0) return false
        if (sign == 0) sign = s else if (s != sign) return false
    }
    return true
}

private fun angleCos(prev: Point, p: Point, next: Point): Double {
    val dx1 = prev.x - p.x; val dy1 = prev.y - p.y
    val dx2 = next.x - p.x; val dy2 = next.y - p.y
    return (dx1 * dx2 + dy1 * dy2) / (sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)) + 1e-10)
}

fun polygonArea(pts: List<Point>): Double {
    var area = 0.0
    for (i in pts.indices) {
        val p1 = pts[i]; val p2 = pts[(i + 1) % pts.size]
        area += p1.x * p2.y - p2.x * p1.y
    }
    return abs(area) / 2.0
}

fun distance(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)

private fun stdDev(mat: Mat): Double {
    val mean = MatOfDouble(); val std = MatOfDouble()
    Core.meanStdDev(mat, mean, std)
    val v = std.toArray().firstOrNull() ?: 0.0
    mean.release(); std.release()
    return v
}

private fun medianOf(mat: Mat): Double {
    val total = mat.rows() * mat.cols()
    val data = ByteArray(total)
    if (mat.isContinuous) {
        mat.get(0, 0, data)
    } else {
        val tmp = mat.clone(); tmp.get(0, 0, data); tmp.release()
    }
    val hist = IntArray(256)
    for (b in data) hist[b.toInt() and 0xFF]++
    var acc = 0
    for (i in 0 until 256) {
        acc += hist[i]
        if (acc >= total / 2) return i.toDouble()
    }
    return 127.0
}

// -------------------------------------------------------------------------
// Kirpma + iyilestirme
// -------------------------------------------------------------------------

/** Dortgeni (TL,TR,BR,BL) dik bir dikdortgene perspektif duzeltmesiyle acar. */
fun cropPicture(picture: Mat, pts: List<Point>): Mat {
    val tl = pts[0]; val tr = pts[1]; val br = pts[2]; val bl = pts[3]
    val dw = max(distance(br, bl), distance(tr, tl))
    val dh = max(distance(tr, br), distance(tl, bl))
    val out = Mat(max(1, dh.toInt()), max(1, dw.toInt()), picture.type())
    val src = MatOfPoint2f(tl, tr, br, bl)
    val dst = MatOfPoint2f(Point(0.0, 0.0), Point(dw, 0.0), Point(dw, dh), Point(0.0, dh))
    val m = Imgproc.getPerspectiveTransform(src, dst)
    Imgproc.warpPerspective(picture, out, m, out.size(), Imgproc.INTER_CUBIC)
    m.release(); src.release(); dst.release()
    return out
}

/** Siyah-beyaz "taranmis belge" gorunumu. */
fun enhancePicture(src: Bitmap): Bitmap {
    val mat = Mat()
    Utils.bitmapToMat(src, mat)
    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)
    var block = (min(mat.cols(), mat.rows()) / 24) or 1
    if (block < 15) block = 15
    Imgproc.adaptiveThreshold(
        mat, mat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, block, 12.0
    )
    val result = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(mat, result)
    mat.release()
    return result
}
