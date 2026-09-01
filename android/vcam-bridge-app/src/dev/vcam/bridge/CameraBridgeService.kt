package dev.vcam.bridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.Log
import android.util.Range
import android.view.Surface
import java.io.IOException
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

class CameraBridgeService : Service() {

    companion object {
        const val ACTION_CONFIGURE = "dev.vcam.bridge.CONFIGURE"
        const val ACTION_START = "dev.vcam.bridge.START"
        const val ACTION_STATUS = "dev.vcam.bridge.STATUS"
        const val ACTION_STOP = "dev.vcam.bridge.STOP"
        const val EXTRA_AE_LOCK = "ae_lock"
        const val EXTRA_AWB_LOCK = "awb_lock"
        const val EXTRA_EXPOSURE_COMPENSATION = "exposure_compensation"
        const val EXTRA_LENS_FACING = "lens_facing"
        const val EXTRA_PERSIST = "persist"
        const val EXTRA_ROTATION = "rotation"
        const val EXTRA_STABILIZATION = "stabilization"
        const val EXTRA_STATUS = "status"
        private const val TAG = "VCamBridge"
        private const val NOTIFICATION_CHANNEL = "vcam_bridge"
        private const val NOTIFICATION_ID = 120
        private const val TCP_PORT = 10020
        private const val WIDTH = 1280
        private const val HEIGHT = 720
        private const val FPS = 30
        private const val BITRATE = 10_000_000
        private const val INITIAL_RETRY_MS = 2_000L
        private const val MAX_RETRY_MS = 60_000L
    }

    private val running = AtomicBoolean(false)
    private var lensFacing = "front"
    private var exposureCompensation = 0
    private var aeLock = false
    private var awbLock = false
    private var stabilization = "video"
    private var rotation = "auto"
    private var effectiveRotationDegrees = 0

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var selectedCameraId: String? = null
    private var cameraOpenPending = false
    private var retryScheduled = false
    private var retryDelayMs = INITIAL_RETRY_MS

    private var encoder: MediaCodec? = null
    private var encoderSurface: Surface? = null
    private var renderer: GlCameraRenderer? = null
    private var serverSocket: ServerSocket? = null
    private var clientSocket: Socket? = null
    private var clientOutput: OutputStream? = null
    private var clientNeedsConfig = true
    private var codecConfig: ByteArray? = null
    private val socketLock = Any()

    private var startedNs = 0L
    private var encodedFrames = 0L
    private var encodedBytes = 0L
    private var capturedFrames = 0L
    private var lastTimestampNs = 0L

    private val cameraCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            capturedFrames++
            lastTimestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L
        }
    }

    override fun onCreate() {
        super.onCreate()
        val prefs = getSharedPreferences("camera", MODE_PRIVATE)
        lensFacing = prefs.getString(EXTRA_LENS_FACING, "front") ?: "front"
        exposureCompensation = prefs.getInt(EXTRA_EXPOSURE_COMPENSATION, 0)
        aeLock = prefs.getBoolean(EXTRA_AE_LOCK, false)
        awbLock = prefs.getBoolean(EXTRA_AWB_LOCK, false)
        stabilization = prefs.getString(EXTRA_STABILIZATION, "video") ?: "video"
        rotation = prefs.getString(EXTRA_ROTATION, "auto") ?: "auto"
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val configuring = intent?.action == ACTION_CONFIGURE
        val lensChanged = configuring && readConfig(intent!!)
        startAsForeground()
        if (!running.get()) {
            try {
                startBridge()
            } catch (e: Exception) {
                Log.e(TAG, "Bridge startup failed", e)
                publishStatus("Bridge startup failed: ${e.message}")
                stopSelf()
            }
        } else if (configuring && cameraHandler != null) {
            cameraHandler?.post {
                if (lensChanged) {
                    closeCamera()
                    openPhysicalCamera()
                } else {
                    updateRendererRotation()
                    submitCaptureRequest()
                }
            }
        }
        return START_STICKY
    }

    private fun readConfig(intent: Intent): Boolean {
        val prev = lensFacing
        if (intent.hasExtra(EXTRA_LENS_FACING)) {
            val req = intent.getStringExtra(EXTRA_LENS_FACING)
            if (req == "front" || req == "back") lensFacing = req
        }
        if (intent.hasExtra(EXTRA_EXPOSURE_COMPENSATION)) {
            exposureCompensation = intent.getIntExtra(EXTRA_EXPOSURE_COMPENSATION, 0).coerceIn(-12, 12)
        }
        if (intent.hasExtra(EXTRA_AE_LOCK)) aeLock = intent.getBooleanExtra(EXTRA_AE_LOCK, false)
        if (intent.hasExtra(EXTRA_AWB_LOCK)) awbLock = intent.getBooleanExtra(EXTRA_AWB_LOCK, false)
        if (intent.hasExtra(EXTRA_STABILIZATION)) {
            val req = intent.getStringExtra(EXTRA_STABILIZATION)
            if (req in setOf("off", "video", "optical")) stabilization = req!!
        }
        if (intent.hasExtra(EXTRA_ROTATION)) {
            rotation = intent.getStringExtra(EXTRA_ROTATION) ?: "auto"
            if (rotation !in setOf("0", "90", "180", "270")) rotation = "auto"
        }
        if (intent.getBooleanExtra(EXTRA_PERSIST, true)) {
            getSharedPreferences("camera", MODE_PRIVATE).edit().apply {
                putString(EXTRA_LENS_FACING, lensFacing)
                putInt(EXTRA_EXPOSURE_COMPENSATION, exposureCompensation)
                putBoolean(EXTRA_AE_LOCK, aeLock)
                putBoolean(EXTRA_AWB_LOCK, awbLock)
                putString(EXTRA_STABILIZATION, stabilization)
                putString(EXTRA_ROTATION, rotation)
                apply()
            }
        }
        return prev != lensFacing
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager?.createNotificationChannel(NotificationChannel(
            NOTIFICATION_CHANNEL, "VCam Bridge", NotificationManager.IMPORTANCE_LOW))
    }

    private fun startAsForeground() {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = Notification.Builder(this, NOTIFICATION_CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("VCam Bridge")
            .setContentText("Streaming physical camera")
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIFICATION_ID, notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun startBridge() {
        running.set(true)
        startedNs = System.nanoTime()
        cameraThread = HandlerThread("vcam-camera").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)
        startTcpServer()
        configureEncoder()
        renderer = GlCameraRenderer(encoderSurface, WIDTH, HEIGHT, WIDTH, HEIGHT).also { it.start() }
        cameraManager = getSystemService(CameraManager::class.java)
        openPhysicalCamera()
        publishStatus("Encoder ready; waiting for local FFmpeg on 127.0.0.1:$TCP_PORT")
    }

    private fun configureEncoder() {
        val format = MediaFormat.createVideoFormat("video/avc", WIDTH, HEIGHT).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, BITRATE)
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR)
            setInteger(MediaFormat.KEY_FRAME_RATE, FPS)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileHigh)
            setInteger(MediaFormat.KEY_LEVEL, MediaCodecInfo.CodecProfileLevel.AVCLevel31)
            if (Build.VERSION.SDK_INT >= 29) setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            if (Build.VERSION.SDK_INT >= 23) {
                setInteger(MediaFormat.KEY_PRIORITY, 0)
                setFloat(MediaFormat.KEY_OPERATING_RATE, FPS.toFloat())
            }
        }
        encoder = MediaCodec.createEncoderByType("video/avc").apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoderSurface = createInputSurface()
            start()
        }
        Thread { drainEncoder() }.also { it.name = "vcam-h264-drain"; it.start() }
        Log.i(TAG, "Encoder=${encoder?.name} ${WIDTH}x${HEIGHT}@${FPS}")
    }

    private fun startTcpServer() {
        Thread {
            try {
                serverSocket = ServerSocket(TCP_PORT, 1, InetAddress.getByName("127.0.0.1"))
                serverSocket?.reuseAddress = true
                while (running.get()) {
                    try {
                        val socket = serverSocket?.accept()
                        socket?.tcpNoDelay = true
                        socket?.sendBufferSize = 262_144
                        synchronized(socketLock) {
                            closeClient()
                            clientSocket = socket
                            clientOutput = socket?.getOutputStream()
                            clientNeedsConfig = true
                        }
                        requestSyncFrame()
                        publishStatus("Physical camera streaming to local FFmpeg at ${WIDTH}x${HEIGHT}@${FPS}")
                    } catch (e: IOException) {
                        if (running.get()) Log.w(TAG, "Client accept failed", e)
                    }
                }
            } catch (e: IOException) {
                Log.e(TAG, "TCP server failed", e)
            }
        }.also { it.name = "vcam-h264-server"; it.start() }
    }

    private fun openPhysicalCamera() {
        if (!running.get() || cameraDevice != null || cameraOpenPending) return
        val manager = cameraManager ?: getSystemService(CameraManager::class.java).also { cameraManager = it }
        try {
            val cameraId = selectPhysicalCamera(manager)
            if (cameraId == null) throw IllegalStateException("no physical camera")
            selectedCameraId = cameraId
            val characteristics = manager.getCameraCharacteristics(cameraId)
            applyRendererRotation(characteristics)
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                throw SecurityException("camera permission revoked")
            }
            cameraOpenPending = true
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenPending = false
                    retryDelayMs = INITIAL_RETRY_MS
                    retryScheduled = false
                    if (!running.get()) { camera.close(); return }
                    cameraDevice = camera
                    createCaptureSession()
                }
                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenPending = false
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    scheduleCameraRetry("Camera disconnected")
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpenPending = false
                    camera.close()
                    if (cameraDevice == camera) cameraDevice = null
                    scheduleCameraRetry("Camera error=$error")
                }
            }, cameraHandler)
        } catch (e: Exception) {
            cameraOpenPending = false
            scheduleCameraRetry("Camera open failed: ${e.message}")
        }
    }

    private fun selectPhysicalCamera(manager: CameraManager): String? {
        val wantFacing = if (lensFacing == "back") CameraCharacteristics.LENS_FACING_BACK else CameraCharacteristics.LENS_FACING_FRONT
        var fallback: String? = null
        for (id in manager.cameraIdList) {
            val facing = manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_EXTERNAL) continue
            if (facing == wantFacing) return id
            if (fallback == null) fallback = id
        }
        return fallback
    }

    private fun createCaptureSession() {
        val camera = cameraDevice ?: return
        val rend = renderer ?: return
        try {
            camera.createCaptureSession(
                listOf(rend.cameraSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        submitCaptureRequest()
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        publishStatus("Capture session config failed")
                    }
                }, cameraHandler)
        } catch (e: CameraAccessException) {
            publishStatus("Capture session failed: ${e.message}")
        }
    }

    private fun submitCaptureRequest() {
        val session = captureSession ?: return
        val camera = cameraDevice ?: return
        try {
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(renderer!!.cameraSurface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    if (stabilization == "video") CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                    else CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AE_LOCK, aeLock)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                set(CaptureRequest.CONTROL_AWB_LOCK, awbLock)
                if (exposureCompensation != 0) set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
                set(CaptureRequest.CONTROL_TARGET_FPS_RANGE, Range(FPS, FPS))
            }
            session.setRepeatingRequest(request.build(), cameraCallback, cameraHandler)
            publishStatus("Camera streaming ($selectedCameraId)")
        } catch (e: CameraAccessException) {
            publishStatus("Capture request failed: ${e.message}")
        }
    }

    private fun applyRendererRotation(characteristics: CameraCharacteristics) {
        val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        effectiveRotationDegrees = when (rotation) {
            "0" -> 0
            "90" -> 90
            "180" -> 180
            "270" -> 270
            else -> sensorOrientation
        }
        renderer?.setRotationQuarterTurns(effectiveRotationDegrees / 90)
    }

    private fun updateRendererRotation() {
        renderer?.setRotationQuarterTurns(effectiveRotationDegrees / 90)
    }

    private fun scheduleCameraRetry(reason: String) {
        Log.w(TAG, "$reason; retrying in ${retryDelayMs}ms")
        publishStatus("$reason; retrying in ${retryDelayMs}ms")
        if (!retryScheduled) {
            retryScheduled = true
            cameraHandler?.postDelayed({ retryScheduled = false; openPhysicalCamera() }, retryDelayMs)
            retryDelayMs = (retryDelayMs * 2).coerceAtMost(MAX_RETRY_MS)
        }
    }

    private fun closeCamera() {
        captureSession?.close(); captureSession = null
        cameraDevice?.close(); cameraDevice = null
    }

    private fun closeClient() {
        try { clientSocket?.close() } catch (_: IOException) {}
        clientSocket = null
        clientOutput = null
    }

    private fun drainEncoder() {
        val info = MediaCodec.BufferInfo()
        while (running.get()) {
            try {
                val index = encoder?.dequeueOutputBuffer(info, 100_000) ?: continue
                if (index >= 0) {
                    val buffer = encoder?.getOutputBuffer(index)
                    if (buffer != null && info.size > 0) {
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            codecConfig = ByteArray(info.size).also { buffer.get(it) }
                        } else {
                            val frame = ByteArray(info.size)
                            buffer.get(frame)
                            writeFrame(frame, info.presentationTimeUs)
                            encodedFrames++
                            encodedBytes += info.size
                        }
                        encoder?.releaseOutputBuffer(index, false)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Encoder drain error", e)
                break
            }
        }
    }

    private fun writeFrame(frame: ByteArray, timestampUs: Long) {
        synchronized(socketLock) {
            val out = clientOutput ?: return
            try {
                if (clientNeedsConfig && codecConfig != null) {
                    out.write(codecConfig)
                    out.flush()
                    clientNeedsConfig = false
                }
                out.write(frame)
                out.flush()
            } catch (e: IOException) {
                Log.w(TAG, "Client write failed", e)
                closeClient()
            }
        }
    }

    private fun requestSyncFrame() {}

    private fun publishStatus(status: String) {
        Log.i(TAG, status)
        val intent = Intent(ACTION_STATUS).apply {
            setPackage(packageName)
            putExtra(EXTRA_STATUS, status)
        }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        running.set(false)
        closeCamera()
        renderer?.close()
        encoder?.stop(); encoder?.release(); encoder = null
        closeClient()
        serverSocket?.close()
        cameraThread?.quitSafely()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
