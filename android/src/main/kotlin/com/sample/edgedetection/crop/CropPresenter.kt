package com.sample.edgedetection.crop

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import com.sample.edgedetection.EdgeDetectionHandler
import com.sample.edgedetection.SourceManager
import com.sample.edgedetection.processor.Corners
import com.sample.edgedetection.processor.TAG
import com.sample.edgedetection.processor.cropPicture
import com.sample.edgedetection.processor.enhancePicture
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

/**
 * Kirpma ekraninin mantigi. Durum basittir: kirpilmis renkli goruntu + "siyah-beyaz mi"
 * + toplam donus acisi. Gosterilen ve kaydedilen goruntu her zaman bu durumdan
 * yeniden uretilir; boylece dondur / S-B / sifirla siralamasindan bagimsiz tutarlidir.
 *
 * [SourceManager.pic] RGBA bir Mat'tir.
 */
class CropPresenter(
    private val iCropView: ICropView.Proxy,
    private val initialBundle: Bundle
) {
    private val picture: Mat? = SourceManager.pic
    private val corners: Corners? = SourceManager.corners

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    private var cropped: Bitmap? = null
    private var rendered: Bitmap? = null
    private var blackWhite = false
    private var rotation = 0
    private var working = false

    val hasPicture: Boolean get() = picture != null && !picture.empty()

    fun onViewsReady() {
        val pic = picture ?: return
        val bitmap = Bitmap.createBitmap(pic.width(), pic.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(pic, bitmap)
        iCropView.getPaper().setImageBitmap(bitmap)
        iCropView.getPaperRect().startCrop(
            corners?.corners?.filterNotNull()?.takeIf { it.size == 4 },
            pic.width(),
            pic.height()
        )
    }

    fun crop(onDone: () -> Unit) {
        val pic = picture ?: return
        if (working || cropped != null) return
        val quad = iCropView.getPaperRect().getCropCorners()
        if (quad.size != 4) return
        working = true
        iCropView.setBusy(true)
        worker.execute {
            val result: Bitmap? = try {
                val mat = cropPicture(pic, quad)
                val bmp = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(mat, bmp)
                mat.release()
                bmp
            } catch (e: Throwable) {
                Log.e(TAG, "Kirpma hatasi", e)
                null
            }
            main.post {
                working = false
                iCropView.setBusy(false)
                if (result != null) {
                    cropped = result
                    rendered = result
                    iCropView.getCroppedPaper().setImageBitmap(result)
                    iCropView.getCroppedPaper().visibility = View.VISIBLE
                    iCropView.getPaper().visibility = View.GONE
                    iCropView.getPaperRect().visibility = View.GONE
                    onDone()
                }
            }
        }
    }

    fun enhance() {
        if (cropped == null || blackWhite) return
        blackWhite = true
        render()
    }

    fun reset() {
        if (cropped == null) return
        blackWhite = false
        rotation = 0
        render()
    }

    fun rotate() {
        if (cropped == null) return
        rotation = (rotation + 270) % 360 // her basista 90 derece saat yonu tersine
        render()
    }

    private fun render() {
        val src = cropped ?: return
        val bw = blackWhite
        val deg = rotation
        worker.execute {
            val out = buildOutput(src, bw, deg)
            main.post {
                rendered = out
                iCropView.getCroppedPaper().setImageBitmap(out)
            }
        }
    }

    private fun buildOutput(src: Bitmap, bw: Boolean, deg: Int): Bitmap {
        var out = if (bw) enhancePicture(src) else src
        if (deg != 0) {
            val m = Matrix().apply { postRotate(deg.toFloat()) }
            out = Bitmap.createBitmap(out, 0, 0, out.width, out.height, m, true)
        }
        return out
    }

    /** Sonucu arka planda JPEG olarak kaydeder; [onDone] ana thread'de cagrilir. */
    fun save(onDone: (Boolean) -> Unit) {
        val src = cropped
        val path = initialBundle.getString(EdgeDetectionHandler.SAVE_TO)
        if (src == null || path.isNullOrEmpty()) {
            onDone(false)
            return
        }
        val bw = blackWhite
        val deg = rotation
        iCropView.setBusy(true)
        worker.execute {
            val ok = try {
                val out = buildOutput(src, bw, deg)
                val file = File(path)
                file.parentFile?.mkdirs()
                FileOutputStream(file).use { out.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                true
            } catch (e: Throwable) {
                Log.e(TAG, "Kaydetme hatasi", e)
                false
            }
            main.post {
                iCropView.setBusy(false)
                onDone(ok)
            }
        }
    }

    fun release() {
        worker.shutdown()
    }
}
