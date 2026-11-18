package com.sample.edgedetection

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import com.sample.edgedetection.scan.ScanActivity
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.FlutterPlugin.FlutterPluginBinding
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry

class EdgeDetectionPlugin : FlutterPlugin, ActivityAware {
    private var handler: EdgeDetectionHandler? = null
    private var channel: MethodChannel? = null

    override fun onAttachedToEngine(binding: FlutterPluginBinding) {
        handler = EdgeDetectionHandler()
        channel = MethodChannel(binding.binaryMessenger, "edge_detection").also {
            it.setMethodCallHandler(handler)
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPluginBinding) {
        channel?.setMethodCallHandler(null)
        channel = null
        handler = null
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        handler?.attach(binding)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        handler?.detach()
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        handler?.attach(binding)
    }

    override fun onDetachedFromActivity() {
        handler?.detach()
    }
}

class EdgeDetectionHandler : MethodCallHandler, PluginRegistry.ActivityResultListener {
    private var binding: ActivityPluginBinding? = null
    private var pendingResult: Result? = null

    companion object {
        const val INITIAL_BUNDLE = "initial_bundle"
        const val FROM_GALLERY = "from_gallery"
        const val SAVE_TO = "save_to"
        const val CAN_USE_GALLERY = "can_use_gallery"
        const val SCAN_TITLE = "scan_title"
        const val CROP_TITLE = "crop_title"
        const val CROP_BLACK_WHITE_TITLE = "crop_black_white_title"
        const val CROP_RESET_TITLE = "crop_reset_title"
        // Dart tarafi bunlari zaten gonderiyor (ios_ onekiyle); Android'de
        // otomatik/elle cekim dugmesi icin kullanilir.
        const val AUTO_SCAN_TITLE = "ios_auto_scan_title"
        const val MANUAL_SCAN_TITLE = "ios_manual_scan_title"
    }

    fun attach(binding: ActivityPluginBinding) {
        this.binding?.removeActivityResultListener(this)
        binding.addActivityResultListener(this)
        this.binding = binding
    }

    fun detach() {
        binding?.removeActivityResultListener(this)
        binding = null
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        val activity = binding?.activity
        if (activity == null) {
            result.error("no_activity", "edge_detection plugin requires a foreground activity.", null)
            return
        }
        when (call.method) {
            "edge_detect" -> launch(activity, call, result, fromGallery = false)
            "edge_detect_gallery" -> launch(activity, call, result, fromGallery = true)
            else -> result.notImplemented()
        }
    }

    private fun launch(activity: Activity, call: MethodCall, result: Result, fromGallery: Boolean) {
        if (pendingResult != null) {
            result.error("already_active", "Edge detection is already active", null)
            return
        }
        val saveTo = call.argument<String>(SAVE_TO)
        if (saveTo.isNullOrEmpty()) {
            result.error("invalid_args", "save_to is required", null)
            return
        }
        pendingResult = result

        val bundle = Bundle().apply {
            putString(SAVE_TO, saveTo)
            putString(CROP_TITLE, call.argument<String>(CROP_TITLE) ?: "")
            putString(CROP_BLACK_WHITE_TITLE, call.argument<String>(CROP_BLACK_WHITE_TITLE) ?: "")
            putString(CROP_RESET_TITLE, call.argument<String>(CROP_RESET_TITLE) ?: "")
            if (fromGallery) {
                putBoolean(FROM_GALLERY, true)
            } else {
                putString(SCAN_TITLE, call.argument<String>(SCAN_TITLE) ?: "")
                putBoolean(CAN_USE_GALLERY, call.argument<Boolean>(CAN_USE_GALLERY) ?: false)
                putString(AUTO_SCAN_TITLE, call.argument<String>(AUTO_SCAN_TITLE) ?: "")
                putString(MANUAL_SCAN_TITLE, call.argument<String>(MANUAL_SCAN_TITLE) ?: "")
            }
        }
        val intent = Intent(activity, ScanActivity::class.java).putExtra(INITIAL_BUNDLE, bundle)
        try {
            @Suppress("DEPRECATION")
            activity.startActivityForResult(intent, REQUEST_CODE)
        } catch (e: Exception) {
            finishWithError("launch_failed", e.message ?: "Could not open scanner")
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST_CODE) return false
        when (resultCode) {
            Activity.RESULT_OK -> finishWithSuccess(true)
            ERROR_CODE -> finishWithError(
                ERROR_CODE.toString(),
                data?.getStringExtra("RESULT") ?: "ERROR"
            )
            else -> finishWithSuccess(false)
        }
        return true
    }

    private fun finishWithError(code: String, message: String) {
        val r = pendingResult
        pendingResult = null
        r?.error(code, message, null)
    }

    private fun finishWithSuccess(value: Boolean) {
        val r = pendingResult
        pendingResult = null
        r?.success(value)
    }
}
