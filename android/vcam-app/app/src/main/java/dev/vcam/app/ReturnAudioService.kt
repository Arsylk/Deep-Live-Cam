package dev.vcam.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.SocketException
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Feeds returned webcam PCM into Android's private remote-submix output.
 *
 * This is intentionally not ordinary playback: no sample is ever written
 * unless TYPE_REMOTE_SUBMIX is the selected output device.  The scoped Xposed
 * module selects the paired input only while an app is using the redirected
 * front camera, making the returned webcam microphone look like that app's
 * capture device without leaking it through the speaker.
 */
class ReturnAudioService : Service() {

    companion object {
        const val ACTION_START = "dev.vcam.app.RETURN_AUDIO_START"
        const val ACTION_STOP = "dev.vcam.app.RETURN_AUDIO_STOP"

        private const val TAG = "VCamReturnAudio"
        private const val CHANNEL_ID = "vcam_return_microphone"
        private const val NOTIFICATION_ID = 121
        private const val UDP_PORT = 10025
        private const val SAMPLE_RATE = 48_000
        private const val CHANNELS = 2
        private const val BYTES_PER_FRAME = CHANNELS * 2
        private const val PACKET_BYTES = 4_096
        private const val MAX_PENDING_PACKETS = 20
    }

    private val running = AtomicBoolean(false)
    private val trackLock = Any()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val pending = ArrayDeque<ByteArray>()
    private var audioManager: AudioManager? = null
    private var audioTrack: AudioTrack? = null
    private var unityGainTrack: AudioTrack? = null
    private var unityGainAttempts = 0
    @Volatile private var socket: DatagramSocket? = null
    private var packetsReceived = 0L
    private var bytesWritten = 0L
    private var lastPacketNs = 0L

    private val deviceCallback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>) {
            if (addedDevices.any { it.isRemoteSubmixOutput() }) ensureSubmixTrack()
        }

        override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>) {
            if (removedDevices.any { it.isRemoteSubmixOutput() }) releaseTrack()
        }
    }

    override fun onCreate() {
        super.onCreate()
        audioManager = getSystemService(AudioManager::class.java)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(NOTIFICATION_ID, notification("Waiting for virtual-mic client"))
        if (running.compareAndSet(false, true)) {
            audioManager?.registerAudioDeviceCallback(deviceCallback, null)
            ensureSubmixTrack()
            startUdpReceiver()
            publishStatus("webcam microphone relay listening on localhost:$UDP_PORT")
        }
        return START_STICKY
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java)?.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Returned webcam microphone",
                NotificationManager.IMPORTANCE_LOW
            )
        )
    }

    private fun notification(detail: String): Notification {
        val pi = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Webcam → virtual front microphone")
            .setContentText(detail)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    private fun AudioDeviceInfo.isRemoteSubmixOutput(): Boolean =
        type == AudioDeviceInfo.TYPE_REMOTE_SUBMIX && isSink && address == "0"

    private fun findRemoteSubmixOutput(): AudioDeviceInfo? =
        audioManager?.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
            ?.firstOrNull { it.isRemoteSubmixOutput() }

    private fun ensureSubmixTrack() {
        synchronized(trackLock) {
            if (!running.get() || audioTrack != null) return
            val output = findRemoteSubmixOutput() ?: return
            val minimum = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_STEREO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            if (minimum <= 0) {
                Log.w(TAG, "Remote-submix AudioTrack has invalid minimum buffer $minimum")
                return
            }
            val candidate = try {
                AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                            .build()
                    )
                    .setBufferSizeInBytes(maxOf(minimum, SAMPLE_RATE * BYTES_PER_FRAME / 10))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
            } catch (error: Exception) {
                Log.w(TAG, "Could not create remote-submix AudioTrack", error)
                return
            }
            // Never call play or write unless routing to the private device was
            // accepted. This is the hard guard against phone-speaker leakage.
            if (!candidate.setPreferredDevice(output)) {
                candidate.release()
                Log.w(TAG, "Remote Submix Out rejected as preferred device")
                return
            }
            try {
                candidate.play()
                audioTrack = candidate
                unityGainTrack = null
                unityGainAttempts = 0
                scheduleSubmixUnityGain(candidate)
                getSystemService(NotificationManager::class.java)?.notify(
                    NOTIFICATION_ID,
                    notification("Active only for redirected front-camera recording")
                )
                publishStatus("virtual microphone submix ready (48 kHz stereo)")
                Log.i(TAG, "AudioTrack routed to ${output.productName} address=${output.address}")
            } catch (error: Exception) {
                candidate.release()
                Log.w(TAG, "Could not start remote-submix AudioTrack", error)
            }
        }
    }

    private fun releaseTrack() {
        synchronized(trackLock) {
            val track = audioTrack ?: return
            audioTrack = null
            unityGainTrack = null
            unityGainAttempts = 0
            try { track.pause() } catch (_: Exception) {}
            try { track.flush() } catch (_: Exception) {}
            track.release()
            getSystemService(NotificationManager::class.java)?.notify(
                NOTIFICATION_ID,
                notification("Waiting for virtual-mic client")
            )
            publishStatus("virtual microphone client released; speaker route remains unused")
        }
    }

    /**
     * Remote Submix has its own per-device media-volume index. Leaving it at
     * the user's speaker volume can attenuate a virtual microphone by 30–50
     * dB. Raise only that virtual endpoint after AudioTrack confirms its
     * actual route; speaker and Bluetooth indices are separate and untouched.
     */
    private fun scheduleSubmixUnityGain(track: AudioTrack) {
        mainHandler.postDelayed({ applySubmixUnityGain(track) }, 100)
    }

    private fun applySubmixUnityGain(track: AudioTrack) {
        synchronized(trackLock) {
            if (!running.get() || audioTrack !== track || unityGainTrack === track) return
            val routed = track.routedDevice
            if (routed == null) {
                unityGainAttempts++
                if (unityGainAttempts < 20) scheduleSubmixUnityGain(track)
                else Log.w(TAG, "Remote-submix route did not settle; leaving volume unchanged")
                return
            }
            if (!routed.isRemoteSubmixOutput()) {
                Log.e(TAG, "Refusing virtual-mic gain/write on unexpected route ${routed.type}@${routed.address}")
                releaseTrack()
                return
            }
            val manager = audioManager ?: return
            try {
                val before = manager.getStreamVolume(AudioManager.STREAM_MUSIC)
                val maximum = manager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                manager.setStreamVolume(AudioManager.STREAM_MUSIC, maximum, 0)
                var after = manager.getStreamVolume(AudioManager.STREAM_MUSIC)
                while (after < maximum) {
                    manager.adjustStreamVolume(
                        AudioManager.STREAM_MUSIC,
                        AudioManager.ADJUST_RAISE,
                        0
                    )
                    val next = manager.getStreamVolume(AudioManager.STREAM_MUSIC)
                    if (next <= after) break
                    after = next
                }
                unityGainTrack = track
                Log.i(TAG, "Remote-submix unity gain: media index $before -> $after/$maximum")
                publishStatus("virtual microphone ready at unity gain (48 kHz stereo)")
            } catch (error: Exception) {
                Log.w(TAG, "Could not set Remote Submix unity gain", error)
            }
        }
    }

    private fun startUdpReceiver() {
        Thread {
            val receiveBuffer = ByteArray(PACKET_BYTES)
            try {
                DatagramSocket(null).use { udp ->
                    udp.reuseAddress = true
                    udp.receiveBufferSize = 65_536
                    udp.bind(InetSocketAddress(InetAddress.getByName("127.0.0.1"), UDP_PORT))
                    socket = udp
                    while (running.get()) {
                        val packet = DatagramPacket(receiveBuffer, receiveBuffer.size)
                        udp.receive(packet)
                        if (packet.length <= 0) continue
                        packetsReceived++
                        lastPacketNs = System.nanoTime()
                        deliver(receiveBuffer.copyOf(packet.length))
                    }
                }
            } catch (error: SocketException) {
                if (running.get()) Log.e(TAG, "PCM UDP socket failed", error)
            } catch (error: Exception) {
                if (running.get()) Log.e(TAG, "PCM receiver failed", error)
            } finally {
                socket = null
            }
        }.also {
            it.name = "vcam-return-pcm"
            it.start()
        }
    }

    private fun deliver(bytes: ByteArray) {
        synchronized(trackLock) {
            val track = audioTrack
            if (track == null) {
                pending.addLast(bytes)
                while (pending.size > MAX_PENDING_PACKETS) pending.removeFirst()
                return
            }
            while (pending.isNotEmpty()) {
                if (!writeToTrack(track, pending.removeFirst())) return
            }
            writeToTrack(track, bytes)
        }
    }

    private fun writeToTrack(track: AudioTrack, bytes: ByteArray): Boolean {
        val written = track.write(bytes, 0, bytes.size, AudioTrack.WRITE_BLOCKING)
        if (written > 0) {
            bytesWritten += written
            return true
        }
        Log.w(TAG, "Remote-submix write failed: $written")
        releaseTrack()
        return false
    }

    private fun publishStatus(detail: String) {
        sendBroadcast(Intent(CameraBridgeService.ACTION_STATUS).apply {
            setPackage(packageName)
            putExtra(CameraBridgeService.EXTRA_STATUS, "virtual mic: $detail")
        })
    }

    override fun onDestroy() {
        running.set(false)
        mainHandler.removeCallbacksAndMessages(null)
        socket?.close()
        socket = null
        try { audioManager?.unregisterAudioDeviceCallback(deviceCallback) } catch (_: Exception) {}
        releaseTrack()
        synchronized(trackLock) { pending.clear() }
        Log.i(
            TAG,
            "Stopped packets=$packetsReceived bytesWritten=$bytesWritten " +
                "lastPacketAgeMs=" +
                if (lastPacketNs == 0L) "never" else
                    ((System.nanoTime() - lastPacketNs) / 1_000_000L).toString()
        )
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
