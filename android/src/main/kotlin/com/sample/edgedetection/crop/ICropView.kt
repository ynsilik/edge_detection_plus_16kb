package com.sample.edgedetection.crop

import android.widget.ImageView
import com.sample.edgedetection.view.PaperRectangle

class ICropView {
    interface Proxy {
        fun getPaper(): ImageView
        fun getPaperRect(): PaperRectangle
        fun getCroppedPaper(): ImageView
        fun setBusy(busy: Boolean)
    }
}
