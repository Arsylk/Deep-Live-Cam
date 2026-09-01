package dev.vcam.bridge

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Main UI: bridge start/stop + camera list + live preview toggle.
 * Merges the old bridge app and camera2-test app into one UI.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusView: TextView
    private lateinit var logView: TextView
    private lateinit var camListView: TextView
    private lateinit var cameraManager: CameraManager
    private val logLines = ArrayDeque<String>()
    private val maxLogLines = 300

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val status = intent.getStringExtra(CameraBridgeService.EXTRA_STATUS)
            if (status != null) log("bridge: $status")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        checkPermissionsAndStart()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
            setBackgroundColor(0xFF0A0E14.toInt())
        }

        val title = TextView(this).apply {
            text = "VCam Bridge"
            textSize = 22f
            setTextColor(0xFF7EE787.toInt())
        }
        root.addView(title)

        statusView = TextView(this).apply {
            textSize = 14f
            setTextColor(0xFFD7E0EA.toInt())
            setPadding(8, 16, 8, 12)
            text = "Ready"
        }
        root.addView(statusView)

        val btnRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        btnRow.addView(Button(this).apply {
            text = "Start Bridge"
            setOnClickListener { startBridge() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        btnRow.addView(Button(this).apply {
            text = "Stop Bridge"
            setOnClickListener { stopBridge() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        btnRow.addView(Button(this).apply {
            text = "Preview"
            setOnClickListener { startActivity(Intent(this@MainActivity, PreviewActivity::class.java)) }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(btnRow)

        val camLabel = TextView(this).apply {
            textSize = 13f
            setTextColor(0xFF9DACBB.toInt())
            setPadding(0, 12, 0, 4)
            text = "Cameras:"
        }
        root.addView(camLabel)
        camListView = TextView(this).apply {
            textSize = 12f
            setTextColor(0xFF8B949E.toInt())
            setPadding(0, 0, 0, 12)
        }
        root.addView(camListView)
        refreshCameraList()

        val refreshBtn = Button(this).apply {
            text = "Refresh Cameras"
            setOnClickListener { refreshCameraList() }
        }
        root.addView(refreshBtn)

        logView = TextView(this).apply {
            textSize = 11f
            setTextColor(0xFF79C0FF.toInt())
            setPadding(8, 8, 8, 8)
            setBackgroundColor(0xFF161B22.toInt())
        }
        root.addView(ScrollView(this).apply { addView(logView) },
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
    }

    private fun refreshCameraList() {
        try {
            val sb = StringBuilder()
            for (id in cameraManager.cameraIdList) {
                val cc = cameraManager.getCameraCharacteristics(id)
                val facing = when (cc.get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_FRONT -> "front"
                    CameraCharacteristics.LENS_FACING_BACK -> "back"
                    CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                    else -> "?"
                }
                val level = when (cc.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)) {
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "level_3"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "full"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "limited"
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "legacy"
                    else -> "?"
                }
                val orient = cc.get(CameraCharacteristics.SENSOR_ORIENTATION)
                sb.appendLine("  $id: facing=$facing level=$level orient=$orient")
            }
            camListView.text = sb.toString()
        } catch (e: Exception) {
            camListView.text = "Error: ${e.message}"
        }
    }

    private fun startBridge() {
        val intent = Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(intent) else startService(intent)
        log("bridge: starting...")
    }

    private fun stopBridge() {
        startService(Intent(this, CameraBridgeService::class.java).apply {
            action = CameraBridgeService.ACTION_STOP
        })
        log("bridge: stopped")
    }

    private fun log(msg: String) {
        logLines.addLast(msg)
        while (logLines.size > maxLogLines) logLines.removeFirst()
        logView.text = logLines.joinToString("\n")
    }

    private fun checkPermissionsAndStart() {
        val perms = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) perms.add(Manifest.permission.POST_NOTIFICATIONS)
        val missing = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 100)
        } else {
            startBridge()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == 100 && results.isNotEmpty()
            && results[0] == PackageManager.PERMISSION_GRANTED) {
            startBridge()
        }
    }

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
