package com.sample.edgedetection.scan

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.HapticFeedbackConstants
import android.view.MenuItem
import android.view.MotionEvent
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import com.sample.edgedetection.ERROR_CODE
import com.sample.edgedetection.EdgeDetectionHandler
import com.sample.edgedetection.R
import com.sample.edgedetection.REQUEST_CODE
import com.sample.edgedetection.SourceManager
import com.sample.edgedetection.base.BaseActivity
import com.sample.edgedetection.crop.CropActivity
import com.sample.edgedetection.processor.Corners
import com.sample.edgedetection.processor.PREVIEW_WORK_SIZE
import com.sample.edgedetection.processor.QuadStabilizer
import com.sample.edgedetection.processor.meanCornerDistance
import com.sample.edgedetection.processor.refineCorners
import com.sample.edgedetection.processor.processPicture
import com.sample.edgedetection.view.PaperRectangle
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.hypot
import kotlin.math.max

/**
 * Kamera ile kart/dokuman tarama ekrani (CameraX).
 *
 *  - Onizleme, analiz ve cekim ayni 4:3 sensor alanini kullanir; boylece
 *    onizlemede bulunan dortgen cekilen fotografa birebir tasinabilir.
 *  - Analiz ~640x480 RGBA karelerde arka planda calisir; sonuc [QuadStabilizer]
 *    ile yumusatilir ve dortgen yeterince sabit kalinca otomatik cekim yapilir.
 *  - Cekilen fotografta tespit daha yuksek cozunurlukte tekrarlanir; onizlemedeki
 *    dortgen ipucu olarak verilir (bulunamazsa dogrudan o kullanilir).
 *  - [EdgeDetectionHandler.FROM_GALLERY] ile acilirsa kamera hic baslatilmaz,
 *    sistem fotograf seciciyle (izin gerektirmez) galeri akisi calisir.
 */
@androidx.annotation.OptIn(markerClass = [TransformExperimental::class])
class ScanActivity : BaseActivity() {

    private lateinit var initialBundle: Bundle
    private lateinit var previewView: PreviewView
    private lateinit var overlay: PaperRectangle
    private lateinit var shutter: View
    private lateinit var flashButton: View
    private lateinit var captureFlash: View
    private lateinit var busy: View
    private lateinit var autoToggle: TextView

    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null

    private val stabilizer = QuadStabilizer() // yalnizca ana thread
    @Volatile private var hint: List<Point>? = null
    @Volatile private var capturing = false
    private var analysisW = 0
    private var analysisH = 0
    private var torchOn = false
    private var autoCapture = true
    private var fromGallery = false
    private var opencvReady = false

    // Analiz / cekim / onizleme arasindaki gercek sensor eslemesi. Akislarin gorus
    // alanlari cihaza gore farkli kirpilabilir; "hepsi 4:3, ayni alan" varsayimi
    // dortgenin kaymasina yol aciyordu.
    private val transformFactory = ImageProxyTransformFactory().apply { isUsingRotationDegrees = true }
    private var lastAnalysisTransform: OutputTransform? = null // ana thread

    // Analiz icin tekrar kullanilan tampon (GC baskisini azaltir).
    private var rowBuffer: ByteArray? = null

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else finishWithError("camera_permission_denied")
        }

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) {
                onImagePicked(uri)
            } else if (fromGallery) {
                setResult(Activity.RESULT_CANCELED)
                finish()
            } else {
                resumeScanning()
            }
        }

    override fun provideContentViewId(): Int = R.layout.activity_scan

    override fun initPresenter() {}

    override fun prepare() {
        initialBundle = intent.getBundleExtra(EdgeDetectionHandler.INITIAL_BUNDLE) ?: Bundle()
        fromGallery = initialBundle.getBoolean(EdgeDetectionHandler.FROM_GALLERY, false)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.paper_rect)
        shutter = findViewById(R.id.shut)
        flashButton = findViewById(R.id.flash)
        captureFlash = findViewById(R.id.capture_flash)
        busy = findViewById(R.id.busy)
        autoToggle = findViewById(R.id.auto_toggle)

        opencvReady = OpenCVLoader.initDebug()
        if (!opencvReady) {
            finishWithError("opencv_load_failed")
            return
        }

        if (fromGallery) {
            findViewById<View>(R.id.bottom_bar).visibility = View.GONE
            supportActionBar?.hide()
            busy.visibility = View.VISIBLE
            launchPicker()
            return
        }

        title = initialBundle.getString(EdgeDetectionHandler.SCAN_TITLE, "")

        val gallery = findViewById<View>(R.id.gallery)
        gallery.visibility =
            if (initialBundle.getBoolean(EdgeDetectionHandler.CAN_USE_GALLERY, false)) View.VISIBLE
            else View.INVISIBLE
        gallery.setOnClickListener {
            if (!capturing) {
                capturing = true
                launchPicker()
            }
        }

        shutter.setOnClickListener { if (!capturing) capture(auto = false) }
        flashButton.setOnClickListener { toggleTorch() }
        setupAutoToggle()
        setupTapToFocus()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            setResult(Activity.RESULT_CANCELED)
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    // ---------------------------------------------------------------------
    // Kamera
    // ---------------------------------------------------------------------

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                Log.e(TAG, "Kamera saglayicisi alinamadi", e)
                finishWithError("camera_unavailable")
                return@addListener
            }

            val ratio = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build())
                .build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                android.util.Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        )
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(worker) { analyze(it) }

            // ~8MP: OCR icin fazlasiyla yeterli, 50MP sensorlerde bellek tasmasini onler.
            val capture = ImageCapture.Builder()
                // ZSL: tetiklendigi anin karesi halka tampondan alinir; kullanici
                // telefonu oynatsa bile sabitlenen goruntu yakalanir. Desteklenmeyen
                // cihazlarda CameraX otomatik olarak MINIMIZE_LATENCY'ye duser.
                .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
                .setJpegQuality(95)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                android.util.Size(3264, 2448),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                            )
                        )
                        .build()
                )
                .build()
            imageCapture = capture

            try {
                provider.unbindAll()
                val cam = provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture
                )
                camera = cam
                Log.i(TAG, "ZSL destegi: ${cam.cameraInfo.isZslSupported}")
                flashButton.visibility =
                    if (cam.cameraInfo.hasFlashUnit()) View.VISIBLE else View.INVISIBLE
            } catch (e: Exception) {
                Log.e(TAG, "Kamera baglanamadi", e)
                finishWithError("camera_unavailable")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleTorch() {
        val cam = camera ?: return
        torchOn = !torchOn
        cam.cameraControl.enableTorch(torchOn)
        flashButton.alpha = if (torchOn) 1f else 0.7f
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupTapToFocus() {
        previewView.setOnTouchListener { v, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                val cam = camera
                if (cam != null) {
                    val point = previewView.meteringPointFactory.createPoint(event.x, event.y)
                    val action = FocusMeteringAction.Builder(point)
                        .setAutoCancelDuration(3, TimeUnit.SECONDS)
                        .build()
                    cam.cameraControl.startFocusAndMetering(action)
                }
                v.performClick()
            }
            true
        }
    }

    private fun setupAutoToggle() {
        val autoTitle = initialBundle.getString(EdgeDetectionHandler.AUTO_SCAN_TITLE).orEmpty()
        val manualTitle = initialBundle.getString(EdgeDetectionHandler.MANUAL_SCAN_TITLE).orEmpty()
        if (autoTitle.isEmpty() || manualTitle.isEmpty()) return
        autoToggle.visibility = View.VISIBLE
        autoToggle.text = autoTitle
        autoToggle.setOnClickListener {
            autoCapture = !autoCapture
            autoToggle.text = if (autoCapture) autoTitle else manualTitle
        }
    }

    // ---------------------------------------------------------------------
    // Canli analiz (worker thread)
    // ---------------------------------------------------------------------

    private fun analyze(image: ImageProxy) {
        if (capturing) {
            image.close()
            return
        }
        var srcTransform: OutputTransform? = null
        val frame: Mat = try {
            srcTransform = transformFactory.getOutputTransform(image)
            rgbaMat(image).also { rotateInPlace(it, image.imageInfo.rotationDegrees) }
        } catch (e: Exception) {
            Log.e(TAG, "Kare donusturulemedi", e)
            null
        } finally {
            image.close()
        } ?: return

        val w = frame.cols()
        val h = frame.rows()
        maybeDumpFrame(frame)
        val result = processPicture(frame, hint, PREVIEW_WORK_SIZE)
        frame.release()
        val corners = result?.corners?.filterNotNull()?.takeIf { it.size == 4 }

        val st = srcTransform
        runOnUiThread { onFrameAnalyzed(corners, w, h, st) }
    }

    private fun onFrameAnalyzed(corners: List<Point>?, w: Int, h: Int, src: OutputTransform?) {
        if (capturing || isFinishing) return
        analysisW = w
        analysisH = h
        lastAnalysisTransform = src
        val now = SystemClock.elapsedRealtime()
        if (corners != null) {
            stabilizer.onDetected(corners, hypot(w.toDouble(), h.toDouble()), now)
        } else {
            stabilizer.onMissed()
        }
        val quad = stabilizer.quad
        hint = quad
        val progress = if (autoCapture) stabilizer.stabilityProgress(now) else 0f
        val viewTarget = previewView.outputTransform
        if (quad != null && src != null && viewTarget != null) {
            val pts = FloatArray(8) { i -> if (i % 2 == 0) quad[i / 2].x.toFloat() else quad[i / 2].y.toFloat() }
            CoordinateTransform(src, viewTarget).mapPoints(pts)
            overlay.showScanQuadInView(pts, progress)
        } else {
            overlay.showScanQuad(quad, w, h, progress)
        }

        if (autoCapture && stabilizer.isReadyForCapture(now)) {
            capture(auto = true)
        }
    }

    /**
     * Yalnizca debug derlemelerde: cache/dump_frames dosyasi varsa her ~2 sn'de bir
     * analiz karesini cache/edge_frame.png olarak yazar (tespit hatalarini
     * masaustunde ayni algoritmayla incelemek icin).
     */
    private var lastDumpMs = 0L
    private fun maybeDumpFrame(frame: Mat) {
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastDumpMs < 2000) return
        lastDumpMs = now
        if (!java.io.File(cacheDir, "dump_frames").exists()) return
        val bgr = Mat()
        Imgproc.cvtColor(frame, bgr, Imgproc.COLOR_RGBA2BGR)
        Imgcodecs.imwrite(java.io.File(cacheDir, "edge_frame.png").absolutePath, bgr)
        bgr.release()
    }

    private fun rgbaMat(image: ImageProxy): Mat {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val w = image.width
        val h = image.height
        val rowStride = plane.rowStride
        val rowBytes = w * 4
        val mat = Mat(h, w, CvType.CV_8UC4)
        buffer.rewind()
        if (rowStride == rowBytes) {
            val size = rowBytes * h
            val data = rowBuffer?.takeIf { it.size == size } ?: ByteArray(size).also { rowBuffer = it }
            buffer.get(data, 0, size)
            mat.put(0, 0, data)
        } else {
            val row = ByteArray(rowBytes)
            for (y in 0 until h) {
                buffer.position(y * rowStride)
                buffer.get(row, 0, rowBytes)
                mat.put(y, 0, row)
            }
        }
        return mat
    }

    private fun rotateInPlace(mat: Mat, degrees: Int) {
        when ((degrees % 360 + 360) % 360) {
            90 -> Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(mat, mat, Core.ROTATE_180)
            270 -> Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE)
        }
    }

    // ---------------------------------------------------------------------
    // Cekim
    // ---------------------------------------------------------------------

    private fun capture(auto: Boolean) {
        val ic = imageCapture ?: return
        if (capturing) return
        capturing = true
        shutter.isEnabled = false
        captureFlash.animate().cancel()
        captureFlash.alpha = 0.85f
        captureFlash.animate().alpha(0f).setDuration(250).start()
        Log.i(TAG, if (auto) "Otomatik cekim" else "Elle cekim")

        // Onizleme dortgenini oransal koordinatlara cevir (cekim ayni FOV'a sahip).
        val q = stabilizer.quad
        val srcT = lastAnalysisTransform
        val fallbackHint = if (q != null && analysisW > 0 && analysisH > 0) {
            q.map { Point(it.x / analysisW, it.y / analysisH) }
        } else null

        var normHint: List<Point>? = fallbackHint
        ic.takePicture(worker, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                // Geri bildirim kare alindiktan SONRA: titresim telefonu oynatmasin.
                runOnUiThread {
                    overlay.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                    busy.visibility = View.VISIBLE
                }
                val mat: Mat? = try {
                    val rotation = image.imageInfo.rotationDegrees
                    normHint = mapToCapture(q, srcT, image, rotation) ?: fallbackHint
                    val bmp = image.toBitmap()
                    val m = bitmapToRgbaMat(bmp, rotation)
                    bmp.recycle()
                    m
                } catch (e: Throwable) {
                    Log.e(TAG, "Cekilen goruntu islenemedi", e)
                    null
                } finally {
                    image.close()
                }
                if (mat == null) {
                    runOnUiThread { onCaptureFailed() }
                } else {
                    handOffToCrop(mat, normHint)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "Cekim hatasi", exception)
                runOnUiThread { onCaptureFailed() }
            }
        })
    }

    /**
     * Onizleme dortgenini (analiz karesi koordinatlari) cekilen fotografin dik
     * goruntusune sensor-dogru eslemeyle tasir; 0..1 oransal koordinat dondurur.
     */
    private fun mapToCapture(
        q: List<Point>?, srcT: OutputTransform?, image: ImageProxy, rotation: Int
    ): List<Point>? {
        if (q == null || srcT == null) return null
        return try {
            val capT = transformFactory.getOutputTransform(image)
            val pts = FloatArray(8) { i -> if (i % 2 == 0) q[i / 2].x.toFloat() else q[i / 2].y.toFloat() }
            CoordinateTransform(srcT, capT).mapPoints(pts)
            val upW = if (rotation % 180 == 0) image.width else image.height
            val upH = if (rotation % 180 == 0) image.height else image.width
            val out = List(4) { Point(pts[it * 2].toDouble() / upW, pts[it * 2 + 1].toDouble() / upH) }
            // Akil kontrolu: esleme bozuksa (or. cihaz donusum bilgisi vermiyorsa) kullanma.
            if (out.any { it.x < -0.2 || it.y < -0.2 || it.x > 1.2 || it.y > 1.2 }) null else out
        } catch (e: Exception) {
            Log.w(TAG, "Cekim eslemesi yapilamadi", e)
            null
        }
    }

    private fun onCaptureFailed() {
        Toast.makeText(this, "Capture failed, please try again", Toast.LENGTH_SHORT).show()
        resumeScanning()
    }

    private fun resumeScanning() {
        capturing = false
        shutter.isEnabled = true
        busy.visibility = View.GONE
        stabilizer.reset()
        hint = null
        overlay.showScanQuad(null, 0, 0, 0f)
    }

    /** Worker thread'de calisir: yuksek cozunurlukte tespit + kirpma ekranina gecis. */
    private fun handOffToCrop(pic: Mat, normHint: List<Point>?) {
        val w = pic.cols().toDouble()
        val h = pic.rows().toDouble()
        val stillHint = normHint?.map { Point(it.x * w, it.y * h) }
        // 1) Tespit, canli akisla AYNI olcekte (kararli parametreler).
        val detected = processPicture(pic, stillHint, PREVIEW_WORK_SIZE, hintTolerance = 0.25)
            ?.corners?.filterNotNull()?.takeIf { it.size == 4 }
        // 2) ZSL ile fotograf, sabitlenen onizleme kareyle ayni an. Onizleme dortgeni
        //    1+ sn boyunca dogrulandigi icin guvenilirdir; fotograftaki tespit ondan
        //    belirgin saparsa onizleme dortgeni esas alinir.
        val diag = hypot(w, h)
        val base = when {
            detected != null && (stillHint == null ||
                meanCornerDistance(detected, stillHint, diag) < 0.04) -> detected
            stillHint != null -> stillHint
            else -> detected
        }
        // 3) Hassasiyet: yuksek cozunurlukte gradyan tabanli kose iyilestirme.
        val corners = base?.let { refineCorners(pic, it) }
        Log.i(TAG, "Cekim: tespit=${detected != null} ipucu=${stillHint != null} " +
            "kullanilan=${if (base === detected) "tespit" else "onizleme"}")

        SourceManager.pic?.release()
        SourceManager.pic = pic
        SourceManager.corners = corners?.let { Corners(it, pic.size()) }

        runOnUiThread {
            if (isFinishing) return@runOnUiThread
            busy.visibility = View.GONE
            val cropIntent = Intent(this, CropActivity::class.java)
                .putExtra(EdgeDetectionHandler.INITIAL_BUNDLE, initialBundle)
            @Suppress("DEPRECATION")
            startActivityForResult(cropIntent, REQUEST_CODE)
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_CODE) return
        when {
            resultCode == Activity.RESULT_OK -> {
                setResult(Activity.RESULT_OK)
                finish()
            }
            resultCode == ERROR_CODE -> {
                setResult(ERROR_CODE, data)
                finish()
            }
            fromGallery -> {
                setResult(Activity.RESULT_CANCELED)
                finish()
            }
            else -> resumeScanning() // kirpma ekranindan geri -> yeniden cek
        }
    }

    // ---------------------------------------------------------------------
    // Galeri
    // ---------------------------------------------------------------------

    private fun launchPicker() {
        pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    private fun onImagePicked(uri: Uri) {
        capturing = true
        busy.visibility = View.VISIBLE
        worker.execute {
            val mat = try {
                decodeUri(uri)
            } catch (e: Throwable) {
                Log.e(TAG, "Galeri goruntusu okunamadi", e)
                null
            }
            if (mat == null) {
                runOnUiThread {
                    if (fromGallery) finishWithError("image_decode_failed") else onCaptureFailed()
                }
            } else {
                handOffToCrop(mat, null)
            }
        }
    }

    /** Uri'yi EXIF yonune gore dondurulmus, en fazla ~3264px RGBA Mat olarak cozer. */
    private fun decodeUri(uri: Uri): Mat? {
        val resolver = contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_STILL_SIDE) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: return null

        val rotation = try {
            resolver.openInputStream(uri)?.use {
                when (ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270
                    else -> 0
                }
            } ?: 0
        } catch (e: Exception) {
            0
        }
        val mat = bitmapToRgbaMat(bmp, rotation)
        bmp.recycle()
        return mat
    }

    private fun bitmapToRgbaMat(src: Bitmap, rotation: Int): Mat {
        var bmp = if (src.config != Bitmap.Config.ARGB_8888) src.copy(Bitmap.Config.ARGB_8888, false) else src
        val longSide = max(bmp.width, bmp.height)
        if (longSide > MAX_STILL_SIDE) {
            val s = MAX_STILL_SIDE.toFloat() / longSide
            val m = Matrix().apply { postScale(s, s) }
            val scaled = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            if (bmp !== src) bmp.recycle()
            bmp = scaled
        }
        val mat = Mat()
        Utils.bitmapToMat(bmp, mat) // RGBA
        if (bmp !== src) bmp.recycle()
        rotateInPlace(mat, rotation)
        return mat
    }

    private fun finishWithError(message: String) {
        val data = Intent().putExtra("RESULT", message)
        setResult(ERROR_CODE, data)
        finish()
    }

    companion object {
        private const val MAX_STILL_SIDE = 3264
    }
}
