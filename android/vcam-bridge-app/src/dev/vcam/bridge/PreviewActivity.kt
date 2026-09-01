package dev.vcam.bridge

import android.hardware.camera2.CameraCharacteristics
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
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

/**
 * Live camera preview with real-time frame analysis.
 * Opens camera 101 (or any front camera), shows the feed,
 * reports luma/FPS/frame stats.
 */
class PreviewActivity : AppCompatActivity() {

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
    private lateinit var previewView: SurfaceView
    private var previewSurface: Surface? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var imageReader: ImageReader? = null
    private var frameCount = 0L
    private var firstFrameNs = 0L
    private var lastFrameNs = 0L
    private var destroying = false

    private val frameWatchdog = object : Runnable {
        override fun run() {
            if (cameraDevice != null && lastFrameNs > 0) {
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
        if (!destroying) maybeOpenCamera()
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
        previewView = SurfaceView(this).apply {
            holder.setFixedSize(WIDTH, HEIGHT)
            holder.addCallback(object : SurfaceHolder.Callback {
                override fun surfaceCreated(holder: SurfaceHolder) {
                    previewSurface = holder.surface
                    maybeOpenCamera()
                }
                override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, h2: Int) {
                    previewSurface = h.surface
                }
                override fun surfaceDestroyed(holder: SurfaceHolder) {
                    previewSurface = null
                    closeCamera()
                }
            })
        }
        root.addView(previewView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    private fun maybeOpenCamera() {
        if (destroying || previewSurface == null || cameraDevice != null) return
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            val ids = manager.cameraIdList
            var selected: String? = null
            for (id in ids) { if (id == "101") { selected = id; break } }
            if (selected == null) {
                for (id in ids) {
                    val cc = manager.getCameraCharacteristics(id)
                    if (cc.get(CameraCharacteristics.LENS_FACING) ==
                        CameraCharacteristics.LENS_FACING_FRONT) {
                        selected = id; break
                    }
                }
            }
            if (selected == null) {
                report("FAIL: No camera found")
                return
            }
            val sel = selected
            report("Opening camera $sel...")
            manager.openCamera(sel, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (destroying || previewSurface == null) { camera.close(); return }
                    cameraDevice = camera
                    startPreview(camera, sel)
                }
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    report("Camera $sel disconnected")
                    scheduleRecovery("disconnected")
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    report("Camera $sel error=$error")
                    scheduleRecovery("error $error")
                }
            }, cameraHandler)
        } catch (e: Exception) {
            report("FAIL: ${e.message}")
            scheduleRecovery("open failed")
        }
    }

    private fun startPreview(camera: CameraDevice, cameraId: String) {
        if (previewSurface == null || !previewSurface!!.isValid) return
        imageReader = ImageReader.newInstance(WIDTH, HEIGHT,
            android.graphics.ImageFormat.YUV_420_888, 3)
        imageReader!!.setOnImageAvailableListener({ reader ->
            reader.acquireLatestImage()?.use { image -> analyzeFrame(cameraId, image) }
        }, cameraHandler)
        try {
            camera.createCaptureSession(
                listOf(previewSurface!!, imageReader!!.surface),
                object : android.hardware.camera2.CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: android.hardware.camera2.CameraCaptureSession) {
                        try {
                            val req = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                                addTarget(previewSurface!!)
                                addTarget(imageReader!!.surface)
                                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                            }
                            session.setRepeatingRequest(req.build(), null, cameraHandler)
                            report("Camera $cameraId streaming...")
                        } catch (e: Exception) {
                            report("Repeating request failed: ${e.message}")
                        }
                    }
                    override fun onConfigureFailed(session: android.hardware.camera2.CameraCaptureSession) {
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
            val rowStart = y.position() + row * rowStride
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
        val elapsed = maxOf(0.001, (now - firstFrameNs) / 1e9)
        val fps = if (frameCount > 1) (frameCount - 1) / elapsed else 0.0
        val mean = if (samples > 0) total.toDouble() / samples else 0.0
        val stats = String.format(Locale.US,
            "PASS: Camera %s | Frames: %d | FPS: %.1f | Luma: min=%d mean=%.2f max=%d",
            cameraId, frameCount, fps, minimum, mean, maximum)
        if (frameCount == 1L || frameCount % 30 == 0L) Log.i(TAG, stats)
        if (frameCount == 1L || frameCount % 10 == 0L) report(stats)
    }

    private fun report(msg: String) {
        Log.i(TAG, msg.replace('\n', ' '))
        runOnUiThread { statusView.text = msg }
    }

    private fun scheduleRecovery(reason: String) {
        mainHandler.postDelayed(cameraRecovery, RECOVERY_DELAY_MS)
    }

    private fun closeCamera() {
        imageReader?.close(); imageReader = null
        cameraDevice?.close(); cameraDevice = null
        frameCount = 0; firstFrameNs = 0; lastFrameNs = 0
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
