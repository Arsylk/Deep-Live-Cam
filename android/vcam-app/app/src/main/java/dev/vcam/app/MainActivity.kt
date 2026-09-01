package dev.vcam.app

import android.annotation.SuppressLint
import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.app.Activity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main UI for the merged vcam app.
 *
 * Live dashboard (per-second telemetry from the bridge service), rolling
 * timestamped event log, bridge start/stop, live preview, and stream
 * configuration (front/back lens selection, rotation).
 */
class MainActivity : Activity() {

    private lateinit var dashView: TextView
    private lateinit var logView: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var frontBtn: Button
    private lateinit var backBtn: Button
    private lateinit var rotBtn: Button
    private lateinit var cameraManager: CameraManager

    private var currentLens = "front"
    private var currentRotation = "auto"
    private val rotations = listOf("auto", "0", "90", "180", "270")

    private val logLines = ArrayDeque<String>()
    private val maxLogLines = 400
    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.US)

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val telemetry = intent.getStringExtra(CameraBridgeService.EXTRA_TELEMETRY)
            if (telemetry != null) {
                dashView.text = telemetry
                return
            }
            val status = intent.getStringExtra(CameraBridgeService.EXTRA_STATUS)
            if (status != null) log("bridge: $status")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = getSharedPreferences("camera", MODE_PRIVATE)
        currentLens = prefs.getString(CameraBridgeService.EXTRA_LENS_FACING, "front") ?: "front"
        currentRotation = prefs.getString(CameraBridgeService.EXTRA_ROTATION, "auto") ?: "auto"
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        buildUi()
        startReturnAudioRelay()
        checkPermissionsAndStart()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 40, 20, 20)
            setBackgroundColor(0xFF0A0E14.toInt())
        }

        val title = TextView(this).apply {
            text = "VCam: Physical Input → Processed Return"
            textSize = 20f
            setTextColor(0xFF7EE787.toInt())
        }
        root.addView(title)

        // Live dashboard (per-second telemetry)
        dashView = TextView(this).apply {
            textSize = 10.5f
            typeface = Typeface.MONOSPACE
            setTextColor(0xFFE3B341.toInt())
            setPadding(10, 10, 10, 10)
            setBackgroundColor(0xFF161B22.toInt())
            text = "Physical input bridge starting; Preview opens processed Camera2 ID 120"
        }
        root.addView(dashView)

        // Row 1: bridge controls
        val row1 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row1.addView(Button(this).apply {
            text = "Start Bridge"
            setOnClickListener { startBridge() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row1.addView(Button(this).apply {
            text = "Stop Bridge"
            setOnClickListener { stopBridge() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row1.addView(Button(this).apply {
            text = "Preview"
            setOnClickListener { togglePreview() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(row1)

        // Row 2: stream source selector + rotation
        val row2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        frontBtn = Button(this).apply {
            text = "Stream: FRONT"
            setOnClickListener { setLens("front") }
        }
        row2.addView(frontBtn, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        backBtn = Button(this).apply {
            text = "Stream: BACK"
            setOnClickListener { setLens("back") }
        }
        row2.addView(backBtn, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        rotBtn = Button(this).apply {
            text = "Rot: $currentRotation"
            setOnClickListener { cycleRotation() }
        }
        row2.addView(rotBtn, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(row2)
        updateLensButtons()

        // Camera list
        val camLabel = TextView(this).apply {
            textSize = 12f
            setTextColor(0xFF9DACBB.toInt())
            setPadding(0, 10, 0, 2)
            text = "Camera2 devices (120 = processed return):"
        }
        root.addView(camLabel)
        val camList = TextView(this).apply {
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setTextColor(0xFF8B949E.toInt())
        }
        root.addView(camList)
        val refreshBtn = Button(this).apply {
            text = "Refresh Camera List"
            setOnClickListener { refreshCameraList(camList) }
        }
        root.addView(refreshBtn)
        refreshCameraList(camList)

        // Rolling event log
        val logLabel = TextView(this).apply {
            textSize = 12f
            setTextColor(0xFF9DACBB.toInt())
            setPadding(0, 10, 0, 2)
            text = "Event log:"
        }
        root.addView(logLabel)
        logView = TextView(this).apply {
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(0xFF79C0FF.toInt())
            setPadding(8, 8, 8, 8)
            setBackgroundColor(0xFF161B22.toInt())
        }
        logScroll = ScrollView(this).apply { addView(logView) }
        root.addView(logScroll, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
    }

    private fun updateLensButtons() {
        frontBtn.text = if (currentLens == "front") ">> FRONT <<" else "Stream: FRONT"
        backBtn.text = if (currentLens == "back") ">> BACK <<" else "Stream: BACK"
        rotBtn.text = "Rot: $currentRotation"
    }

    private fun setLens(facing: String) {
        currentLens = facing
        updateLensButtons()
        log("ui: stream lens -> $facing")
        startService(Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_CONFIGURE
            putExtra(CameraBridgeService.EXTRA_LENS_FACING, facing)
        })
    }

    private fun cycleRotation() {
        val idx = rotations.indexOf(currentRotation)
        currentRotation = rotations[(idx + 1) % rotations.size]
        updateLensButtons()
        log("ui: rotation -> $currentRotation")
        startService(Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_CONFIGURE
            putExtra(CameraBridgeService.EXTRA_ROTATION, currentRotation)
        })
    }

    private fun refreshCameraList(camList: TextView) {
        try {
            val ids = cameraManager.cameraIdList
            val sb = StringBuilder()
            for (id in ids) {
                val cc = cameraManager.getCameraCharacteristics(id)
                val facing = when (cc.get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_FRONT -> "front"
                    CameraCharacteristics.LENS_FACING_BACK -> "back"
                    CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                    else -> "?"
                }
                val level = when (cc.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)) {
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "legacy"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "limited"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "full"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "external"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "level_3"
                    else -> "?"
                }
                val orientation = cc.get(CameraCharacteristics.SENSOR_ORIENTATION)
                val route = if (id == "120") "PROCESSED RETURN" else "physical/input candidate"
                sb.appendLine("  $id [$route]: facing=$facing level=$level orient=$orientation")
            }
            camList.text = sb.toString()
        } catch (e: Exception) {
            camList.text = "Error: ${e.message}"
        }
    }

    private fun startBridge() {
        startReturnAudioRelay()
        val intent = Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        log("ui: bridge starting...")
    }

    private fun startReturnAudioRelay() {
        val intent = Intent(this, ReturnAudioService::class.java).apply {
            action = ReturnAudioService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }

    private fun stopBridge() {
        startService(Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_STOP
        })
        log("ui: bridge stopped")
    }

    private fun togglePreview() {
        try {
            startActivity(Intent(this, PreviewActivity::class.java))
        } catch (e: Exception) {
            log("preview unavailable: ${e.message}")
        }
    }

    private fun log(msg: String) {
        logLines.addLast("${timeFmt.format(Date())} $msg")
        while (logLines.size > maxLogLines) logLines.removeFirst()
        logView.text = logLines.joinToString("\n")
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun checkPermissionsAndStart() {
        val perms = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) {
            perms.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val missing = perms.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 100)
        } else {
            log("ui: ready; tap Start to send the phone camera to Arch")
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == 100 && results.isNotEmpty()
            && results[0] == PackageManager.PERMISSION_GRANTED
        ) {
            log("ui: camera permission granted; tap Start when phone input is wanted")
        }
    }

    @SuppressLint("UnspecifiedRegisterReceiverFlag")
    override fun onStart() {
        super.onStart()
        val filter = IntentFilter(CameraBridgeService.ACTION_STATUS)
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(statusReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(statusReceiver, filter)
        }
    }

    override fun onStop() {
        unregisterReceiver(statusReceiver)
        super.onStop()
    }
}
