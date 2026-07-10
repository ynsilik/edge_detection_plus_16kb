import 'dart:async';

import 'package:flutter/services.dart';

class EdgeDetectionPlus {
  static const MethodChannel _channel = const MethodChannel('edge_detection');

  /// Call this method to scan the object edge in live camera.
  static Future<bool> detectEdge(
    String saveTo, {
    bool canUseGallery = true,
    String androidScanTitle = "Scanning",
    String androidCropTitle = "Crop",
    String androidCropBlackWhiteTitle = "Black White",
    String androidCropReset = "Reset",
    String iosCancelTitle = "Cancel",
    String iosAutoScanTitle = "Auto",
    String iosManualScanTitle = "Manual",
    String iosNextTitle = "Next",
    String iosEditTitle = "Edit Scan",
    String iosReviewTitle = "Review",
  }) async {
    return await _channel.invokeMethod('edge_detect', {
      'save_to': saveTo,
      'can_use_gallery': canUseGallery,
      'scan_title': androidScanTitle,
      'crop_title': androidCropTitle,
      'crop_black_white_title': androidCropBlackWhiteTitle,
      'crop_reset_title': androidCropReset,
      'ios_cancel_title': iosCancelTitle,
      'ios_auto_scan_title': iosAutoScanTitle,
      'ios_manual_scan_title': iosManualScanTitle,
      'ios_next_title': iosNextTitle,
      'ios_edit_title': iosEditTitle,
      'ios_review_title': iosReviewTitle,
    });
  }

  /// Call this method to scan the object edge from a gallery image.
  static Future<bool> detectEdgeFromGallery(
    String saveTo, {
    String androidCropTitle = "Crop",
    String androidCropBlackWhiteTitle = "Black White",
    String androidCropReset = "Reset",
    String iosCancelTitle = "Cancel",
    String iosNextTitle = "Next",
    String iosEditTitle = "Edit Scan",
    String iosReviewTitle = "Review",
  }) async {
    return await _channel.invokeMethod('edge_detect_gallery', {
      'save_to': saveTo,
      'crop_title': androidCropTitle,
      'crop_black_white_title': androidCropBlackWhiteTitle,
      'crop_reset_title': androidCropReset,
      'from_gallery': true,
      'ios_cancel_title': iosCancelTitle,
      'ios_next_title': iosNextTitle,
      'ios_edit_title': iosEditTitle,
      'ios_review_title': iosReviewTitle,
    });
  }
}
