package dev.vcam.app

import android.app.Activity
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.Locale

/**
 * Processed-return preview with real-time frame analysis.
 * Opens external Camera2 ID 120 first, or another external camera as a
 * compatibility fallback. It never consumes the physical input camera.
 */
class PreviewActivity : Activity() {

    companion object {
        private const val TAG = "VCamPreview"
        private const val WIDTH = 1280
        private const val HEIGHT = 720
        private const val LUMA_SAMPLE_STEP = 8
        private const val STALL_TIMEOUT_NS = 2_500_000_000L
        private const val RECOVERY_DELAY_MS = 1_000L
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var statusView: TextView
    private lateinit var textureView: TextureView
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var imageReader: ImageReader? = null
    private var frameCount = 0L
    private var firstFrameNs = 0L
    private var lastFrameNs = 0L
    private var destroying = false
    private var started = false
    private var streamSize = Size(WIDTH, HEIGHT)

    private val frameWatchdog = object : Runnable {
        override fun run() {
            if (started && cameraDevice != null && lastFrameNs > 0) {
                val ageNs = System.nanoTime() - lastFrameNs
                if (ageNs > STALL_TIMEOUT_NS) {
                    report("WAITING: No fresh frames (stalled)")
                    scheduleRecovery("stream stalled")
                }
            }
            mainHandler.postDelayed(this, 1000)
        }
    }

    private val cameraRecovery = Runnable {
        if (started && !destroying) maybeOpenCamera()
    }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
            st.setDefaultBufferSize(WIDTH, HEIGHT)
            maybeOpenCamera()
        }
        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {
            st.setDefaultBufferSize(WIDTH, HEIGHT)
            configureTransform(w, h)
        }
        override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
            closeCamera()
            return true
        }
        override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        cameraThread = HandlerThread("vcam-preview").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)
        mainHandler.post(frameWatchdog)
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 32, 16, 16)
            setBackgroundColor(0xFF0A0E14.toInt())
        }
        statusView = TextView(this).apply {
            textSize = 13f
            setTextColor(0xFFD7E0EA.toInt())
            setPadding(8, 8, 8, 8)
            text = "Opening camera..."
        }
        root.addView(statusView)
        textureView = TextureView(this)
        textureView.surfaceTextureListener = surfaceListener
        root.addView(textureView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    /**
     * Aspect-correct cover transform. The stream is landscape (1280x720) with
     * the subject already upright, so we do NOT rotate. We center-crop
     * horizontally to fill the portrait view without stretching and without
     * black bars.
     */
    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        if (viewWidth == 0 || viewHeight == 0) return
        val sw = streamSize.width.toFloat()
        val sh = streamSize.height.toFloat()
        // TextureView stretches the buffer to fill the view by default
        // (anisotropic scaleX/scaleY). Compute the correction to a uniform
        // cover scale so the image keeps its true aspect ratio.
        val stretchX = viewWidth / sw
        val stretchY = viewHeight / sh
        val cover = Math.max(viewWidth / sw, viewHeight / sh)
        val m = Matrix()
        m.setScale(cover / stretchX, cover / stretchY)
        val scaledW = sw * cover
        val scaledH = sh * cover
        m.postTranslate((viewWidth - scaledW) / 2f, (viewHeight - scaledH) / 2f)
        textureView.setTransform(m)
    }

    private fun maybeOpenCamera() {
        if (!started || destroying || !textureView.isAvailable || cameraDevice != null) return
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            val ids = manager.cameraIdList
            val selected = ids.firstOrNull { it == "120" } ?: ids.firstOrNull { id ->
                manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_EXTERNAL
            }
            if (selected == null) {
                report("FAIL: Processed return camera 120 is unavailable (no external fallback)")
                return
            }
            if (checkSelfPermission(android.Manifest.permission.CAMERA) !=
                android.content.pm.PackageManager.PERMISSION_GRANTED
            ) {
                report("FAIL: Camera permission is not granted")
                return
            }
            report("Opening processed return camera $selected...")
            manager.openCamera(selected, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (!started || destroying || !textureView.isAvailable) {
                        camera.close()
                        return
                    }
                    cameraDevice = camera
                    startPreview(camera, selected)
                }
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    report("Processed return camera $selected disconnected")
                    scheduleRecovery("disconnected")
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    report("Processed return camera $selected error=$error")
                    scheduleRecovery("error $error")
                }
            }, cameraHandler)
        } catch (e: Exception) {
            report("FAIL: ${e.message}")
            scheduleRecovery("open failed")
        }
    }

    private fun startPreview(camera: CameraDevice, cameraId: String) {
        if (!textureView.isAvailable) return
        textureView.surfaceTexture?.setDefaultBufferSize(WIDTH, HEIGHT)
        imageReader = ImageReader.newInstance(WIDTH, HEIGHT, ImageFormat.YUV_420_888, 3)
        imageReader!!.setOnImageAvailableListener({ reader ->
            reader.acquireLatestImage()?.use { img -> analyzeFrame(cameraId, img) }
        }, cameraHandler)
        val previewSurface = Surface(textureView.surfaceTexture)
        try {
            camera.createCaptureSession(
                listOf(previewSurface, imageReader!!.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        try {
                            val req = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                                addTarget(previewSurface)
                                addTarget(imageReader!!.surface)
                                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                            }
                            session.setRepeatingRequest(req.build(), null, cameraHandler)
                            runOnUiThread {
                                configureTransform(textureView.width, textureView.height)
                            }
                            report("Processed return camera $cameraId streaming...")
                        } catch (e: Exception) {
                            report("Repeating request failed: ${e.message}")
                        }
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        report("Session config failed")
                        scheduleRecovery("config failed")
                    }
                }, cameraHandler)
        } catch (e: Exception) {
            report("Session failed: ${e.message}")
            scheduleRecovery("session failed")
        }
    }

    private fun analyzeFrame(cameraId: String, image: Image) {
        val y = image.planes[0].buffer
        val rowStride = image.planes[0].rowStride
        val pixStride = image.planes[0].pixelStride
        var total = 0L; var minimum = 255; var maximum = 0; var samples = 0
        for (row in 0 until image.height step LUMA_SAMPLE_STEP) {
            val rowStart = row * rowStride
            for (col in 0 until image.width step LUMA_SAMPLE_STEP) {
                val off = rowStart + col * pixStride
                if (off >= y.limit()) break
                val v = y.get(off).toInt() and 0xff
                total += v
                minimum = minOf(minimum, v)
                maximum = maxOf(maximum, v)
                samples++
            }
        }
        frameCount++
        val now = System.nanoTime()
        lastFrameNs = now
        if (firstFrameNs == 0L) firstFrameNs = now
        val elapsed = Math.max(0.001, (now - firstFrameNs) / 1e9)
        val fps = if (frameCount > 1) (frameCount - 1) / elapsed else 0.0
        val mean = if (samples > 0) total.toDouble() / samples else 0.0
        val stats = String.format(Locale.US,
            "PASS: Processed return %s | Frames: %d | FPS: %.1f | Luma: min=%d mean=%.2f max=%d",
            cameraId, frameCount, fps, minimum, mean, maximum)
        if (frameCount == 1L || frameCount % 30 == 0L) Log.i(TAG, stats)
        if (frameCount == 1L || frameCount % 10 == 0L) report(stats)
    }

    private fun report(msg: String) {
        Log.i(TAG, msg.replace('\n', ' '))
        runOnUiThread { statusView.text = msg }
    }

    private fun scheduleRecovery(@Suppress("UNUSED_PARAMETER") reason: String) {
        if (!started || destroying) return
        mainHandler.removeCallbacks(cameraRecovery)
        mainHandler.postDelayed(cameraRecovery, RECOVERY_DELAY_MS)
    }

    private fun closeCamera() {
        mainHandler.removeCallbacks(cameraRecovery)
        imageReader?.close(); imageReader = null
        cameraDevice?.close(); cameraDevice = null
        frameCount = 0; firstFrameNs = 0; lastFrameNs = 0
    }

    override fun onStart() {
        super.onStart()
        started = true
        maybeOpenCamera()
    }

    override fun onStop() {
        started = false
        closeCamera()
        super.onStop()
    }

    override fun onDestroy() {
        destroying = true
        mainHandler.removeCallbacks(frameWatchdog)
        mainHandler.removeCallbacks(cameraRecovery)
        closeCamera()
        cameraThread?.quitSafely()
        super.onDestroy()
    }
}
