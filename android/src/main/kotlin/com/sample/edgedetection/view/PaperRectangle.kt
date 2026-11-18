package com.sample.edgedetection.view

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.Build
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.widget.Magnifier
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Iki modlu cizim katmani:
 *  - Tarama modu: kamera onizlemesi uzerinde bulunan dortgeni gosterir. Onizleme
 *    FILL_CENTER oldugu icin goruntu->ekran donusumu "kapla + ortala" ile yapilir.
 *    Kareler arasi gecis animasyonla yumusatilir; sabitlik ilerlemesi renkle
 *    (beyaz -> yesil) gosterilir.
 *  - Kirpma modu: FIT_CENTER gosterilen fotograf uzerinde surukelenebilir 4 kose,
 *    Android 9+ cihazlarda buyutec ile.
 *
 * Disaridan gelen noktalar hicbir zaman yerinde degistirilmez.
 */
class PaperRectangle @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private val density = resources.displayMetrics.density

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeJoin = Paint.Join.ROUND
        color = Color.WHITE
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(110, 0, 0, 0)
    }
    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(70, 255, 255, 255)
    }
    private val handleStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        color = Color.WHITE
    }
    private val path = Path()
    private val outerPath = Path()

    // Goruntu -> view donusumu
    private var imgW = 0.0
    private var imgH = 0.0
    private var scale = 1.0
    private var offX = 0.0
    private var offY = 0.0
    private var cropMode = false

    // View koordinatlarinda: hedef ve ekranda cizilen (animasyonlu) koseler
    private var target: Array<FloatArray>? = null
    private var shown: Array<FloatArray>? = null
    private var progress = 0f

    // Kirpma modu
    private var activeCorner = -1
    private var magnifier: Any? = null

    // ---------------------------------------------------------------------
    // Tarama modu
    // ---------------------------------------------------------------------

    /** [corners] goruntu koordinatlarinda (TL,TR,BR,BL); null -> gizle. */
    fun showScanQuad(corners: List<Point>?, imageWidth: Int, imageHeight: Int, stability: Float) {
        cropMode = false
        progress = stability
        if (corners == null || corners.size != 4 || width == 0 || height == 0) {
            if (target != null) {
                target = null
                shown = null
                invalidate()
            }
            return
        }
        setTransform(imageWidth.toDouble(), imageHeight.toDouble(), fill = true)
        val t = Array(4) { toView(corners[it]) }
        target = t
        if (shown == null) shown = Array(4) { t[it].copyOf() }
        postInvalidateOnAnimation()
    }

    /**
     * [viewPoints] dogrudan bu view'in koordinatlarinda (x0,y0,...,x3,y3); CameraX
     * CoordinateTransform ile hesaplanmis olmali. null -> gizle.
     */
    fun showScanQuadInView(viewPoints: FloatArray?, stability: Float) {
        cropMode = false
        progress = stability
        if (viewPoints == null || viewPoints.size != 8) {
            if (target != null) {
                target = null
                shown = null
                invalidate()
            }
            return
        }
        val t = Array(4) { floatArrayOf(viewPoints[it * 2], viewPoints[it * 2 + 1]) }
        target = t
        if (shown == null) shown = Array(4) { t[it].copyOf() }
        postInvalidateOnAnimation()
    }

    // ---------------------------------------------------------------------
    // Kirpma modu
    // ---------------------------------------------------------------------

    /** Kirpma ekranini baslatir; [corners] null ise varsayilan cerceve kullanilir. */
    fun startCrop(corners: List<Point>?, imageWidth: Int, imageHeight: Int) {
        cropMode = true
        progress = 0f
        setTransform(imageWidth.toDouble(), imageHeight.toDouble(), fill = false)
        val pts = if (corners != null && corners.size == 4) corners else listOf(
            Point(imageWidth * 0.1, imageHeight * 0.1),
            Point(imageWidth * 0.9, imageHeight * 0.1),
            Point(imageWidth * 0.9, imageHeight * 0.9),
            Point(imageWidth * 0.1, imageHeight * 0.9)
        )
        val t = Array(4) { clampToImage(toView(pts[it])) }
        target = t
        shown = Array(4) { t[it].copyOf() }
        invalidate()
    }

    /** Kirpma modundaki koseleri goruntu koordinatlarinda dondurur. */
    fun getCropCorners(): List<Point> {
        val s = shown ?: return emptyList()
        return List(4) { toImage(s[it]) }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!cropMode) return false
        val s = shown ?: return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val touchRadius = 48f * density
                var best = -1
                var bestD = Float.MAX_VALUE
                for (i in 0 until 4) {
                    val d = hypot(s[i][0] - event.x, s[i][1] - event.y)
                    if (d < bestD) { bestD = d; best = i }
                }
                activeCorner = if (bestD <= touchRadius) best else -1
                if (activeCorner >= 0) {
                    parent?.requestDisallowInterceptTouchEvent(true)
                    showMagnifier(s[activeCorner][0], s[activeCorner][1])
                    invalidate()
                }
                return activeCorner >= 0
            }
            MotionEvent.ACTION_MOVE -> {
                if (activeCorner < 0) return false
                val p = clampToImage(floatArrayOf(event.x, event.y))
                val old = s[activeCorner]
                s[activeCorner] = p
                if (!isConvex(s)) s[activeCorner] = old // kendini kesen dortgene izin verme
                showMagnifier(s[activeCorner][0], s[activeCorner][1])
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                activeCorner = -1
                hideMagnifier()
                invalidate()
                return true
            }
        }
        return false
    }

    // ---------------------------------------------------------------------
    // Cizim
    // ---------------------------------------------------------------------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val t = target ?: return
        val s = shown ?: return

        // Tarama modunda hedefe dogru yumusak gecis.
        if (!cropMode) {
            var moving = false
            for (i in 0 until 4) for (k in 0..1) {
                val diff = t[i][k] - s[i][k]
                if (abs(diff) > 0.5f) {
                    s[i][k] += diff * 0.35f
                    moving = true
                } else {
                    s[i][k] = t[i][k]
                }
            }
            if (moving) postInvalidateOnAnimation()
        }

        path.reset()
        path.moveTo(s[0][0], s[0][1])
        for (i in 1 until 4) path.lineTo(s[i][0], s[i][1])
        path.close()

        if (cropMode) {
            outerPath.reset()
            outerPath.fillType = Path.FillType.EVEN_ODD
            outerPath.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
            outerPath.addPath(path)
            canvas.drawPath(outerPath, dimPaint)
            strokePaint.color = Color.WHITE
            canvas.drawPath(path, strokePaint)
            val r = 14f * density
            for (i in 0 until 4) {
                val rr = if (i == activeCorner) r * 1.4f else r
                canvas.drawCircle(s[i][0], s[i][1], rr, handlePaint)
                canvas.drawCircle(s[i][0], s[i][1], rr, handleStroke)
            }
        } else {
            val color = blend(Color.WHITE, Color.rgb(52, 199, 89), progress)
            fillPaint.color = Color.argb(
                (60 + 50 * progress).toInt(), Color.red(color), Color.green(color), Color.blue(color)
            )
            strokePaint.color = color
            canvas.drawPath(path, fillPaint)
            canvas.drawPath(path, strokePaint)
        }
    }

    // ---------------------------------------------------------------------
    // Yardimcilar
    // ---------------------------------------------------------------------

    private fun setTransform(iw: Double, ih: Double, fill: Boolean) {
        imgW = iw; imgH = ih
        val vw = width.toDouble(); val vh = height.toDouble()
        if (iw <= 0 || ih <= 0 || vw <= 0 || vh <= 0) return
        scale = if (fill) max(vw / iw, vh / ih) else min(vw / iw, vh / ih)
        offX = (vw - iw * scale) / 2.0
        offY = (vh - ih * scale) / 2.0
    }

    private fun toView(p: Point) =
        floatArrayOf((p.x * scale + offX).toFloat(), (p.y * scale + offY).toFloat())

    private fun toImage(p: FloatArray) = Point(
        ((p[0] - offX) / scale).coerceIn(0.0, imgW),
        ((p[1] - offY) / scale).coerceIn(0.0, imgH)
    )

    private fun clampToImage(p: FloatArray): FloatArray {
        val minX = max(0.0, offX).toFloat()
        val maxX = min(width.toDouble(), offX + imgW * scale).toFloat()
        val minY = max(0.0, offY).toFloat()
        val maxY = min(height.toDouble(), offY + imgH * scale).toFloat()
        return floatArrayOf(p[0].coerceIn(minX, maxX), p[1].coerceIn(minY, maxY))
    }

    private fun isConvex(q: Array<FloatArray>): Boolean {
        var sign = 0
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]; val c = q[(i + 2) % 4]
            val cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            val sgn = if (cross > 0) 1 else if (cross < 0) -1 else 0
            if (sgn == 0) return false
            if (sign == 0) sign = sgn else if (sgn != sign) return false
        }
        return true
    }

    private fun showMagnifier(x: Float, y: Float) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return
        val m = (magnifier as? Magnifier) ?: Magnifier(this).also { magnifier = it }
        m.show(x, y)
    }

    private fun hideMagnifier() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return
        (magnifier as? Magnifier)?.dismiss()
    }

    private fun blend(from: Int, to: Int, t: Float): Int {
        val k = t.coerceIn(0f, 1f)
        return Color.rgb(
            (Color.red(from) + (Color.red(to) - Color.red(from)) * k).toInt(),
            (Color.green(from) + (Color.green(to) - Color.green(from)) * k).toInt(),
            (Color.blue(from) + (Color.blue(to) - Color.blue(from)) * k).toInt()
        )
    }
}
