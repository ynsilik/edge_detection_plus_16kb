# OpenCV Java sinirlari JNI uzerinden cagrilir (Java_org_opencv_* sembolleri);
# R8'in bu siniflari/metotlari yeniden adlandirmasi veya silmesi native
# baglamayi bozar ve tarayici release derlemede coker.
-keep class org.opencv.** { *; }
-dontwarn org.opencv.**
