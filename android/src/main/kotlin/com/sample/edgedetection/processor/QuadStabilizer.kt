package com.sample.edgedetection.processor

import org.opencv.core.Point
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Kareler arasi kose titremesini yok eder ve otomatik cekim icin "sabitlik" olcer.
 *
 * Tum mesafeler kare kosegenine oranlanir; cozunurlukten bagimsizdir.
 *  - Ilk tespit ya da buyuk bir sicrama, ancak art arda [CONFIRM_FRAMES] karede
 *    tutarli gorulurse kabul edilir (tek karelik yanlis tespit cizgiyi oynatmaz).
 *  - Kucuk hareketler mesafeye gore uyarlanan EMA ile izlenir; cok kucuk
 *    olanlar (olu bolge) tamamen yutulur, cizgi kipirdamaz.
 *  - Tespit kacirilan karelerde cizgi [MAX_MISSES] kare boyunca korunur.
 *
 * Bu sinifin urettigi noktalar her zaman yeni nesnelerdir; disarida yerinde
 * degistirilse bile ic durum bozulmaz.
 */
class QuadStabilizer {

    companion object {
        private const val CONFIRM_FRAMES = 2
        private const val CONFIRM_DIST = 0.05      // aday dogrulama esigi
        private const val JUMP_DIST = 0.10         // bundan buyuk = baska nesne / sicrama
        private const val DEAD_BAND = 0.004        // bundan kucuk hareket yok sayilir
        private const val STABLE_DIST = 0.02       // ham tespit bu kadar yakinsa "sabit"
        private const val MAX_MISSES = 6
        const val AUTO_CAPTURE_MS = 1100L
        private const val AUTO_CAPTURE_MIN_FRAMES = 6
    }

    /** Ekranda gosterilecek dortgen (kare koordinatlari), yoksa null. */
    var quad: List<Point>? = null
        private set

    private var candidate: List<Point>? = null
    private var candidateHits = 0
    private var misses = 0
    private var stableSinceMs = 0L
    private var stableFrames = 0

    fun reset() {
        quad = null; candidate = null; candidateHits = 0; misses = 0
        stableFrames = 0; stableSinceMs = 0L
    }

    /** Yeni bir tespit isler. [diag] kare kosegeni (px). */
    fun onDetected(raw: List<Point>, diag: Double, nowMs: Long) {
        misses = 0
        val current = quad
        if (current == null) {
            if (confirm(raw, diag)) {
                quad = copy(raw)
                resetStability(nowMs)
            }
            return
        }

        val d = maxDist(current, raw) / diag
        if (d > JUMP_DIST) {
            // Baska bir nesneye atlama ihtimali: dogrulanana kadar eski cizgiyi koru.
            if (confirm(raw, diag)) {
                quad = copy(raw)
                resetStability(nowMs)
            }
            return
        }
        candidate = null; candidateHits = 0

        if (d > DEAD_BAND) {
            // Kucuk hareket -> yumusak takip, buyuk hareket -> hizli takip.
            val alpha = 0.25 + 0.55 * min(1.0, (d - DEAD_BAND) / (JUMP_DIST * 0.6))
            quad = List(4) {
                Point(
                    current[it].x + alpha * (raw[it].x - current[it].x),
                    current[it].y + alpha * (raw[it].y - current[it].y)
                )
            }
        }

        if (d < STABLE_DIST) {
            stableFrames++
        } else {
            resetStability(nowMs)
        }
    }

    /** Bu karede tespit yok. */
    fun onMissed() {
        misses++
        if (misses >= MAX_MISSES) reset()
    }

    /** 0..1 arasi otomatik cekim ilerlemesi. */
    fun stabilityProgress(nowMs: Long): Float {
        if (quad == null || stableSinceMs == 0L) return 0f
        if (misses > 1) return 0f
        return min(1f, max(0f, (nowMs - stableSinceMs).toFloat() / AUTO_CAPTURE_MS))
    }

    fun isReadyForCapture(nowMs: Long): Boolean =
        quad != null && misses == 0 && stableFrames >= AUTO_CAPTURE_MIN_FRAMES &&
            nowMs - stableSinceMs >= AUTO_CAPTURE_MS

    private fun confirm(raw: List<Point>, diag: Double): Boolean {
        val c = candidate
        if (c != null && maxDist(c, raw) / diag < CONFIRM_DIST) {
            candidateHits++
        } else {
            candidateHits = 1
        }
        candidate = copy(raw)
        if (candidateHits >= CONFIRM_FRAMES) {
            candidate = null; candidateHits = 0
            return true
        }
        return false
    }

    private fun resetStability(nowMs: Long) {
        stableFrames = 0
        stableSinceMs = nowMs
    }

    private fun maxDist(a: List<Point>, b: List<Point>): Double {
        var m = 0.0
        for (i in 0 until 4) m = max(m, hypot(a[i].x - b[i].x, a[i].y - b[i].y))
        return m
    }

    private fun copy(src: List<Point>) = src.map { Point(it.x, it.y) }
}
