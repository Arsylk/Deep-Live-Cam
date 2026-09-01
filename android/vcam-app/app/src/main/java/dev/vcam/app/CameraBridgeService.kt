package dev.vcam.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Build
import android.os.Bundle
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
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Camera2 owner and low-latency H.264 producer for the phone bridge. */
class CameraBridgeService : Service() {

    companion object {
        const val ACTION_CONFIGURE = "dev.vcam.app.CONFIGURE"
        const val ACTION_START = "dev.vcam.app.START"
        const val ACTION_STATUS = "dev.vcam.app.STATUS"
        const val ACTION_STOP = "dev.vcam.app.STOP"
        const val EXTRA_AE_LOCK = "ae_lock"
        const val EXTRA_AWB_LOCK = "awb_lock"
        const val EXTRA_EXPOSURE_COMPENSATION = "exposure_compensation"
        const val EXTRA_LENS_FACING = "lens_facing"
        const val EXTRA_PERSIST = "persist"
        const val EXTRA_ROTATION = "rotation"
        const val EXTRA_STABILIZATION = "stabilization"
        const val EXTRA_ZOOM_PERCENT = "zoom_percent"
        const val EXTRA_STATUS = "status"
        const val EXTRA_TELEMETRY = "telemetry"

        private const val TAG = "VCamBridge"
        private const val NOTIFICATION_CHANNEL = "vcam_bridge"
        private const val NOTIFICATION_ID = 120
        private const val TCP_PORT = 10020
        private const val WIDTH = 720
        private const val HEIGHT = 1280
        // The Camera2 stream size: the sensor is natively landscape and
        // offers no portrait stream, so the capture buffer stays at the
        // HAL's native landscape resolution (full field of view). The GL
        // stage rotates it upright into the portrait transport canvas.
        private const val CAMERA_WIDTH = 1280
        private const val CAMERA_HEIGHT = 720
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
    private var stabilization = "off"
    private var rotation = "auto"
    private var zoomPercent = 100
    private var appliedZoomRatio = 1.0f
    private var maximumZoomRatio = 1.0f
    private var effectiveRotationDegrees = 0
    private var sensorOrientationDegrees = 0

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
    @Volatile private var codecConfig: ByteArray? = null
    private val socketLock = Any()

    private var startedNs = 0L
    private var encodedFrames = 0L
    private var encodedBytes = 0L
    private var capturedFrames = 0L
    private var droppedFrames = 0L
    private var lastTimestampNs = 0L
    private var captureIntervalEmaMs = 0.0
    private var captureJitterEmaMs = 0.0

    // Verbose live state for the dashboard.
    private var cameraState = "idle"
    private var encoderState = "stopped"
    private var tcpListening = false
    private var tcpBindAttempts = 0
    private var clientConnectedNs = 0L
    private var clientDisconnectedNs = 0L
    private var clientBytes = 0L
    private var clientWritesFailed = 0L
    private var clientConnectCount = 0
    private var encodedFps = 0.0
    private var capturedFps = 0.0
    private var clientKbps = 0.0
    private var lastTickNs = 0L
    private var lastTickEncoded = 0L
    private var lastTickCaptured = 0L
    private var lastTickClientBytes = 0L
    private val telemetryHandler = Handler(android.os.Looper.getMainLooper())
    private val telemetryTick = object : Runnable {
        override fun run() {
            publishTelemetry()
            telemetryHandler.postDelayed(this, 1000)
        }
    }

    private val cameraCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            recordMetrics(result)
        }
    }

    override fun onCreate() {
        super.onCreate()
        val prefs = getSharedPreferences("camera", MODE_PRIVATE)
        lensFacing = prefs.getString(EXTRA_LENS_FACING, "front") ?: "front"
        exposureCompensation = prefs.getInt(EXTRA_EXPOSURE_COMPENSATION, 0)
        aeLock = prefs.getBoolean(EXTRA_AE_LOCK, false)
        awbLock = prefs.getBoolean(EXTRA_AWB_LOCK, false)
        stabilization = prefs.getString(EXTRA_STABILIZATION, "off") ?: "off"
        rotation = validatedRotation(prefs.getString(EXTRA_ROTATION, "auto"))
        zoomPercent = prefs.getInt(EXTRA_ZOOM_PERCENT, 100).coerceIn(100, 300)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val configuring = intent?.action == ACTION_CONFIGURE
        val lensChanged = configuring && readConfig(intent)
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
            exposureCompensation = intent.getIntExtra(EXTRA_EXPOSURE_COMPENSATION, 0)
                .coerceIn(-12, 12)
        }
        if (intent.hasExtra(EXTRA_AE_LOCK)) aeLock = intent.getBooleanExtra(EXTRA_AE_LOCK, false)
        if (intent.hasExtra(EXTRA_AWB_LOCK)) awbLock = intent.getBooleanExtra(EXTRA_AWB_LOCK, false)
        if (intent.hasExtra(EXTRA_STABILIZATION)) {
            val req = intent.getStringExtra(EXTRA_STABILIZATION)
            if (req == "off" || req == "video" || req == "optical") stabilization = req
        }
        if (intent.hasExtra(EXTRA_ROTATION)) rotation = validatedRotation(intent.getStringExtra(EXTRA_ROTATION))
        if (intent.hasExtra(EXTRA_ZOOM_PERCENT)) {
            zoomPercent = intent.getIntExtra(EXTRA_ZOOM_PERCENT, 100).coerceIn(100, 300)
        }
        if (intent.getBooleanExtra(EXTRA_PERSIST, true)) {
            getSharedPreferences("camera", MODE_PRIVATE).edit().apply {
                putString(EXTRA_LENS_FACING, lensFacing)
                putInt(EXTRA_EXPOSURE_COMPENSATION, exposureCompensation)
                putBoolean(EXTRA_AE_LOCK, aeLock)
                putBoolean(EXTRA_AWB_LOCK, awbLock)
                putString(EXTRA_STABILIZATION, stabilization)
                putString(EXTRA_ROTATION, rotation)
                putInt(EXTRA_ZOOM_PERCENT, zoomPercent)
                apply()
            }
        }
        return prev != lensFacing
    }

    private fun validatedRotation(req: String?): String =
        if (req in setOf("0", "90", "180", "270")) req!! else "auto"

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager?.createNotificationChannel(NotificationChannel(
            NOTIFICATION_CHANNEL, "Deep Live Camera Bridge", NotificationManager.IMPORTANCE_LOW))
    }

    private fun startAsForeground() {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = Notification.Builder(this, NOTIFICATION_CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("Deep Live Camera Bridge")
            .setContentText("Physical input to Arch; processed return is Camera2 ID 120")
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
        renderer = GlCameraRenderer(encoderSurface, WIDTH, HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT).also { it.start() }
        cameraManager = getSystemService(CameraManager::class.java)
        openPhysicalCamera()
        telemetryHandler.removeCallbacks(telemetryTick)
        telemetryHandler.post(telemetryTick)
        publishStatus("Encoder and GPU renderer ready; waiting for local FFmpeg on 127.0.0.1:$TCP_PORT")
    }

    private fun configureEncoder() {
        val format = MediaFormat.createVideoFormat("video/avc", WIDTH, HEIGHT).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, BITRATE)
            setInteger(MediaFormat.KEY_BITRATE_MODE,
                MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR)
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
        encoderState = "started"
        Thread { drainEncoder() }.also { it.name = "vcam-h264-drain"; it.start() }
        Log.i(TAG, "Encoder=${encoder?.name} ${WIDTH}x${HEIGHT}@${FPS} bitrate=$BITRATE")
    }

    private fun startTcpServer() {
        Thread {
            // Bind with backoff so a stale/occupied port never permanently kills the bridge.
            while (running.get() && serverSocket == null) {
                tcpBindAttempts++
                try {
                    val ss = ServerSocket()
                    ss.reuseAddress = true
                    ss.bind(java.net.InetSocketAddress(InetAddress.getByName("127.0.0.1"), TCP_PORT), 1)
                    serverSocket = ss
                    tcpListening = true
                    Log.i(TAG, "TCP bound 127.0.0.1:$TCP_PORT after $tcpBindAttempts attempt(s)")
                } catch (e: IOException) {
                    Log.w(TAG, "TCP bind failed (${e.message}); retrying in 2s [attempt $tcpBindAttempts]")
                    try { Thread.sleep(2000) } catch (_: InterruptedException) {}
                }
            }
            val ss = serverSocket ?: return@Thread
            Log.i(TAG, "TCP server listening on 127.0.0.1:$TCP_PORT")
            publishStatus("TCP listening on 127.0.0.1:$TCP_PORT")
            while (running.get()) {
                try {
                    val socket = ss.accept()
                    socket.tcpNoDelay = true
                    socket.sendBufferSize = 262_144
                    synchronized(socketLock) {
                        closeClient()
                        clientSocket = socket
                        clientOutput = socket.getOutputStream()
                        clientNeedsConfig = true
                        clientConnectedNs = System.nanoTime()
                        clientBytes = 0L
                        clientConnectCount++
                    }
                    requestSyncFrame()
                    publishStatus("FFmpeg client #$clientConnectCount connected from ${socket.remoteSocketAddress}")
                    publishStatus("Physical camera streaming to local FFmpeg at ${WIDTH}x${HEIGHT}@${FPS}")
                } catch (e: IOException) {
                    if (running.get()) Log.w(TAG, "H.264 client accept failed", e)
                }
            }
        }.also { it.name = "vcam-h264-server"; it.start() }
    }

    private fun openPhysicalCamera() {
        if (!running.get() || cameraDevice != null || cameraOpenPending) return
        val manager = cameraManager ?: getSystemService(CameraManager::class.java).also { cameraManager = it }
        try {
            val cameraId = selectPhysicalCamera(manager)
            if (cameraId == null) throw IllegalStateException("no physical camera available")
            selectedCameraId = cameraId
            val characteristics = manager.getCameraCharacteristics(cameraId)
            updateZoomCapabilities(characteristics)
            applyRendererRotation(characteristics)
            if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                throw SecurityException("camera permission revoked")
            }
            cameraOpenPending = true
            cameraState = "opening($cameraId)"
            Log.i(TAG, "openCamera id=$cameraId facing=$lensFacing")
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenPending = false
                    retryDelayMs = INITIAL_RETRY_MS
                    retryScheduled = false
                    if (!running.get()) { camera.close(); return }
                    cameraDevice = camera
                    cameraState = "open($cameraId)"
                    Log.i(TAG, "Camera $cameraId opened; configuring session")
                    createCaptureSession()
                }
                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenPending = false
                    camera.close()
                    if (cameraDevice == camera) { cameraDevice = null; cameraState = "disconnected" }
                    scheduleCameraRetry("Camera disconnected")
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpenPending = false
                    camera.close()
                    if (cameraDevice == camera) { cameraDevice = null; cameraState = "error($error)" }
                    scheduleCameraRetry("Camera error=$error")
                }
            }, cameraHandler)
        } catch (e: Exception) {
            cameraOpenPending = false
            scheduleCameraRetry("Camera open failed: ${e.message}")
        }
    }

    private fun selectPhysicalCamera(manager: CameraManager): String? {
        val wantFacing = if (lensFacing == "back")
            CameraCharacteristics.LENS_FACING_BACK else CameraCharacteristics.LENS_FACING_FRONT
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
                        cameraState = "streaming($selectedCameraId)"
                        Log.i(TAG, "Capture session configured for $selectedCameraId")
                        submitCaptureRequest()
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        cameraState = "session-failed"
                        publishStatus("Capture session configuration failed")
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
            // This Surface ultimately feeds a continuous hardware encoder, so
            // ask the HAL for its recording ISP profile rather than the soft,
            // display-oriented preview profile.
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                addTarget(renderer!!.cameraSurface!!)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    if (stabilization == "video") CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                    else CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AE_LOCK, aeLock)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                set(CaptureRequest.CONTROL_AWB_LOCK, awbLock)
                if (exposureCompensation != 0) {
                    set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
                }
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(FPS, FPS))
                applyZoom(this)
            }
            session.setRepeatingRequest(request.build(), cameraCallback, cameraHandler!!)
            publishStatus("Physical camera streaming ($selectedCameraId)")
        } catch (e: CameraAccessException) {
            publishStatus("Capture request failed: ${e.message}")
        }
    }

    private fun recordMetrics(result: TotalCaptureResult) {
        capturedFrames++
        val timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        val now = System.nanoTime()
        if (lastTimestampNs > 0) {
            val intervalMs = (timestamp - lastTimestampNs) / 1_000_000.0
            val jitterMs = kotlin.math.abs(intervalMs - (1000.0 / FPS))
            captureIntervalEmaMs = if (captureIntervalEmaMs == 0.0) intervalMs
                else captureIntervalEmaMs * 0.95 + intervalMs * 0.05
            captureJitterEmaMs = if (captureJitterEmaMs == 0.0) jitterMs
                else captureJitterEmaMs * 0.95 + jitterMs * 0.05
        }
        lastTimestampNs = timestamp
    }

    private fun applyRendererRotation(characteristics: CameraCharacteristics) {
        val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        sensorOrientationDegrees = sensorOrientation
        updateRendererRotation()
    }

    private fun updateZoomCapabilities(characteristics: CameraCharacteristics) {
        maximumZoomRatio = if (Build.VERSION.SDK_INT >= 30) {
            characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
                ?.upper ?: characteristics.get(
                    CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM
                ) ?: 1.0f
        } else {
            characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
                ?: 1.0f
        }.coerceAtLeast(1.0f)
    }

    /** Apply a live, centred sensor crop without reopening the camera. */
    private fun applyZoom(request: CaptureRequest.Builder) {
        val desired = (zoomPercent / 100.0f).coerceIn(1.0f, maximumZoomRatio)
        appliedZoomRatio = desired
        if (Build.VERSION.SDK_INT >= 30) {
            request.set(CaptureRequest.CONTROL_ZOOM_RATIO, desired)
            return
        }
        val cameraId = selectedCameraId ?: return
        val characteristics = cameraManager?.getCameraCharacteristics(cameraId) ?: return
        val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: return
        val cropWidth = (active.width() / desired).toInt().coerceAtLeast(2)
        val cropHeight = (active.height() / desired).toInt().coerceAtLeast(2)
        val left = active.left + (active.width() - cropWidth) / 2
        val top = active.top + (active.height() - cropHeight) / 2
        request.set(
            CaptureRequest.SCALER_CROP_REGION,
            Rect(left, top, left + cropWidth, top + cropHeight),
        )
    }

    private fun updateRendererRotation() {
        // Recompute this value on every CONFIGURE action. Previously a live
        // rotation change only reapplied the value calculated when the camera
        // was opened, so the control appeared to work but did nothing.
        effectiveRotationDegrees = when (rotation) {
            "0" -> 0
            "90" -> 90
            "180" -> 180
            "270" -> 270
            else -> sensorOrientationDegrees
        }
        renderer?.setRotationQuarterTurns(effectiveRotationDegrees / 90)
    }

    private fun scheduleCameraRetry(reason: String) {
        Log.w(TAG, "$reason; retrying in ${retryDelayMs}ms")
        publishStatus("$reason; retrying in ${retryDelayMs}ms")
        if (!retryScheduled) {
            retryScheduled = true
            cameraHandler?.postDelayed({
                retryScheduled = false
                openPhysicalCamera()
            }, retryDelayMs)
            retryDelayMs = (retryDelayMs * 2).coerceAtMost(MAX_RETRY_MS)
        }
    }

    private fun closeCamera() {
        captureSession?.close(); captureSession = null
        cameraDevice?.close(); cameraDevice = null
        cameraState = "closed"
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
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            cacheCodecConfig(
                                listOf(ByteArray(info.size).also { buffer.get(it) })
                            )
                        } else {
                            val frame = ByteArray(info.size)
                            buffer.get(frame)
                            writeFrame(frame, info.presentationTimeUs)
                            encodedFrames++
                            encodedBytes += info.size
                        }
                        encoder?.releaseOutputBuffer(index, false)
                    } else if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val format = encoder?.outputFormat
                        if (format != null) cacheCodecConfigFromFormat(format)
                        Log.i(TAG, "Encoder format changed: $format")
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
                    Log.i(TAG, "Sent ${codecConfig!!.size}B codec config (SPS/PPS) to client")
                }
                out.write(frame)
                out.flush()
                clientBytes += frame.size
            } catch (e: IOException) {
                clientWritesFailed++
                clientDisconnectedNs = System.nanoTime()
                Log.w(TAG, "Client write failed after ${fmtBytes(clientBytes)} (${clientWritesFailed} errors); client gone", e)
                publishStatus("FFmpeg client disconnected (tx=${fmtBytes(clientBytes)})")
                closeClient()
            }
        }
    }

    private fun cacheCodecConfigFromFormat(format: MediaFormat) {
        val parts = listOf("csd-0", "csd-1").mapNotNull { key ->
            format.getByteBuffer(key)?.duplicate()?.let { buffer ->
                ByteArray(buffer.remaining()).also { buffer.get(it) }
            }
        }
        cacheCodecConfig(parts)
    }

    private fun cacheCodecConfig(parts: List<ByteArray>) {
        val normalized = parts.filter { it.isNotEmpty() }.map { annexB(it) }
        if (normalized.isEmpty()) return
        val combined = ByteArray(normalized.sumOf { it.size })
        var offset = 0
        for (part in normalized) {
            part.copyInto(combined, offset)
            offset += part.size
        }
        codecConfig = combined
        Log.i(TAG, "Cached ${combined.size}B codec config (SPS/PPS)")
    }

    private fun annexB(data: ByteArray): ByteArray {
        val startsWithLongCode = data.size >= 4 &&
            data[0] == 0.toByte() && data[1] == 0.toByte() &&
            data[2] == 0.toByte() && data[3] == 1.toByte()
        val startsWithShortCode = data.size >= 3 &&
            data[0] == 0.toByte() && data[1] == 0.toByte() && data[2] == 1.toByte()
        if (startsWithLongCode || startsWithShortCode) return data
        return byteArrayOf(0, 0, 0, 1) + data
    }

    private fun requestSyncFrame() {
        try {
            encoder?.setParameters(Bundle().apply {
                putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
            })
            Log.i(TAG, "Requested sync frame for newly connected FFmpeg client")
        } catch (e: Exception) {
            // The cached SPS/PPS still makes the next scheduled I-frame
            // decodable on codecs which do not implement this optional hint.
            Log.w(TAG, "Encoder sync-frame request was rejected", e)
        }
    }

    private fun publishStatus(status: String) {
        Log.i(TAG, status)
        val intent = Intent(ACTION_STATUS).apply {
            setPackage(packageName)
            putExtra(EXTRA_STATUS, status)
        }
        sendBroadcast(intent)
    }

    private fun publishTelemetry() {
        val now = System.nanoTime()
        val dt = if (lastTickNs > 0) (now - lastTickNs) / 1e9 else 1.0
        if (dt > 0) {
            encodedFps = (encodedFrames - lastTickEncoded) / dt
            capturedFps = (capturedFrames - lastTickCaptured) / dt
            clientKbps = (clientBytes - lastTickClientBytes) / dt / 1024.0
        }
        lastTickNs = now
        lastTickEncoded = encodedFrames
        lastTickCaptured = capturedFrames
        lastTickClientBytes = clientBytes
        val text = buildTelemetry()
        Log.d(TAG, "telemetry\n$text")
        val intent = Intent(ACTION_STATUS).apply {
            setPackage(packageName)
            putExtra(EXTRA_TELEMETRY, text)
        }
        sendBroadcast(intent)
    }

    private fun buildTelemetry(): String {
        val upS = if (startedNs > 0) (System.nanoTime() - startedNs) / 1e9 else 0.0
        val r = renderer
        val clientUp = if (clientOutput != null && clientConnectedNs > 0)
            "%.0fs".format((System.nanoTime() - clientConnectedNs) / 1e9) else "-"
        val sb = StringBuilder()
        sb.appendLine("up=%.0fs lens=%s rot=%s(%d) stab=%s zoom=%.2fx/%.2fx exp=%+d ae=%s awb=%s"
            .format(upS, lensFacing, rotation, effectiveRotationDegrees, stabilization,
                appliedZoomRatio, maximumZoomRatio, exposureCompensation,
                if (aeLock) "lock" else "auto", if (awbLock) "lock" else "auto"))
        sb.appendLine("cam : id=%s state=%s cap=%d(%.1ffps) int=%.1fms jit=%.2fms"
            .format(selectedCameraId ?: "-", cameraState, capturedFrames, capturedFps,
                captureIntervalEmaMs, captureJitterEmaMs))
        sb.appendLine("enc : %s %s %dx%d@%d frames=%d(%.1ffps) tx=%s"
            .format(encoder?.name ?: "-", encoderState, WIDTH, HEIGHT, FPS, encodedFrames,
                encodedFps, fmtBytes(encodedBytes)))
        sb.appendLine("gl  : egl=%s frames=%d swapErr=%d err=%s"
            .format(if (r?.eglReady == true) "ok" else "-", r?.framesRendered ?: 0,
                r?.swapFailures ?: 0, r?.lastError ?: "-"))
        sb.appendLine("geom: capture=%dx%d framing=cover textureRot=%d shaderRot=%d"
            .format(WIDTH, HEIGHT, r?.textureMatrixRotationDegrees ?: 0,
                r?.shaderRotationDegrees ?: 0))
        sb.appendLine("tcp : :%d listen=%s binds=%d conns=%d client=%s up=%s tx=%s %.0fKB/s wrErr=%d"
            .format(TCP_PORT, if (tcpListening) "yes" else "no", tcpBindAttempts, clientConnectCount,
                if (clientOutput != null) "yes" else "no", clientUp, fmtBytes(clientBytes),
                clientKbps, clientWritesFailed))
        return sb.toString().trimEnd()
    }

    private fun fmtBytes(b: Long): String = when {
        b >= 1_073_741_824 -> "%.2fGB".format(b / 1e9)
        b >= 1_048_576 -> "%.1fMB".format(b / 1e6)
        b >= 1024 -> "%.1fKB".format(b / 1e3)
        else -> "${b}B"
    }

    override fun onDestroy() {
        running.set(false)
        telemetryHandler.removeCallbacks(telemetryTick)
        closeCamera()
        renderer?.close()
        encoder?.stop(); encoder?.release()
        encoder = null
        encoderState = "stopped"
        closeClient()
        serverSocket?.close()
        tcpListening = false
        cameraThread?.quitSafely()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
