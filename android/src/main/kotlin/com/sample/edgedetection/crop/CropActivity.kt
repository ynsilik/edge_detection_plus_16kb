package com.sample.edgedetection.crop

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.ImageView
import com.sample.edgedetection.ERROR_CODE
import com.sample.edgedetection.EdgeDetectionHandler
import com.sample.edgedetection.R
import com.sample.edgedetection.SourceManager
import com.sample.edgedetection.base.BaseActivity
import com.sample.edgedetection.view.PaperRectangle

class CropActivity : BaseActivity(), ICropView.Proxy {

    private var showMenuItems = false
    private lateinit var mPresenter: CropPresenter
    private lateinit var initialBundle: Bundle
    private var saving = false

    override fun provideContentViewId(): Int = R.layout.activity_crop

    override fun initPresenter() {
        initialBundle = intent.getBundleExtra(EdgeDetectionHandler.INITIAL_BUNDLE) ?: Bundle()
        mPresenter = CropPresenter(this, initialBundle)
        findViewById<ImageView>(R.id.crop).setOnClickListener {
            mPresenter.crop { changeMenuVisibility(true) }
        }
    }

    override fun prepare() {
        title = initialBundle.getString(EdgeDetectionHandler.CROP_TITLE).orEmpty()
        if (!mPresenter.hasPicture) {
            setResult(ERROR_CODE, Intent().putExtra("RESULT", "no_picture"))
            finish()
            return
        }
        // Gercek boyutlar ancak yerlesimden sonra belli olur.
        findViewById<View>(R.id.paper_rect).post { mPresenter.onViewsReady() }
    }

    override fun onDestroy() {
        super.onDestroy()
        mPresenter.release()
        if (isFinishing) {
            SourceManager.pic?.release()
            SourceManager.pic = null
            SourceManager.corners = null
        }
    }

    override fun getPaper(): ImageView = findViewById(R.id.paper)
    override fun getPaperRect(): PaperRectangle = findViewById(R.id.paper_rect)
    override fun getCroppedPaper(): ImageView = findViewById(R.id.picture_cropped)
    override fun setBusy(busy: Boolean) {
        findViewById<View>(R.id.crop_busy).visibility = if (busy) View.VISIBLE else View.GONE
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.crop_activity_menu, menu)
        menu.setGroupVisible(R.id.enhance_group, showMenuItems)
        menu.findItem(R.id.rotation_image).isVisible = showMenuItems
        menu.findItem(R.id.action_label).isVisible = showMenuItems
        initialBundle.getString(EdgeDetectionHandler.CROP_BLACK_WHITE_TITLE)
            ?.takeIf { it.isNotEmpty() }?.let { menu.findItem(R.id.gray).title = it }
        initialBundle.getString(EdgeDetectionHandler.CROP_RESET_TITLE)
            ?.takeIf { it.isNotEmpty() }?.let { menu.findItem(R.id.reset).title = it }
        findViewById<View>(R.id.crop_bar).visibility = if (showMenuItems) View.GONE else View.VISIBLE
        return super.onCreateOptionsMenu(menu)
    }

    private fun changeMenuVisibility(show: Boolean) {
        showMenuItems = show
        invalidateOptionsMenu()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> {
                setResult(Activity.RESULT_CANCELED)
                finish()
            }
            R.id.action_label -> {
                if (saving) return true
                saving = true
                item.isEnabled = false
                mPresenter.save { ok ->
                    if (ok) {
                        setResult(Activity.RESULT_OK)
                    } else {
                        setResult(ERROR_CODE, Intent().putExtra("RESULT", "save_failed"))
                    }
                    finish()
                }
            }
            R.id.rotation_image -> mPresenter.rotate()
            R.id.gray -> mPresenter.enhance()
            R.id.reset -> mPresenter.reset()
            else -> return super.onOptionsItemSelected(item)
        }
        return true
    }
}
