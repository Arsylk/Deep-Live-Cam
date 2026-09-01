package dev.vcam.test;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.nio.ByteBuffer;
import java.io.File;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final String TAG = "VCamCamera2Test";
    private static final int CAMERA_PERMISSION_REQUEST = 1;
    private static final String PREFERRED_CAMERA_ID = "101";
    private static final String EXTRA_REQUESTED_CAMERA_ID = "requested_camera_id";
    private static final String EXTRA_STOP_AFTER_FRAMES = "stop_after_frames";
    private static final String EXTRA_AUDIO_MODE = "audio_mode";
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final int LUMA_SAMPLE_STEP = 8;
    private static final long STALL_TIMEOUT_NS = 2_500_000_000L;
    private static final long RECOVERY_DELAY_MS = 1_000L;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private TextView statusView;
    private TextView audioStatusView;
    private TextView previewCover;
    private SurfaceView previewView;
    private Surface previewSurface;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private ImageReader imageReader;
    private boolean surfaceReady;
    private boolean cameraOpenPending;
    private boolean cameraRecoveryScheduled;
    private boolean destroying;
    private int openGeneration;
    private long frameCount;
    private long firstFrameNs;
    private volatile long lastFrameNs;
    private String requestedCameraId;
    private int stopAfterFrames;
    private boolean stopAfterFramesReached;
    private String audioMode;
    private AudioRecord audioRecord;
    private MediaRecorder mediaRecorder;
    private final AtomicBoolean audioRunning = new AtomicBoolean(false);
    private Thread audioThread;
    private File mediaRecorderOutput;

    private final Runnable stopMediaRecorder = this::finishMediaRecorderTest;

    private final Runnable cameraRecovery = new Runnable() {
        @Override
        public void run() {
            cameraRecoveryScheduled = false;
            if (!destroying && surfaceReady) {
                maybeOpenProcessedCamera();
            }
        }
    };

    private final Runnable frameWatchdog = new Runnable() {
        @Override
        public void run() {
            if (captureSession != null) {
                long ageNs = lastFrameNs == 0 ? Long.MAX_VALUE : System.nanoTime() - lastFrameNs;
                if (ageNs > STALL_TIMEOUT_NS) {
                    showPreviewCover("STREAM STALLED\nCamera 120 is open but no fresh frames arrived.");
                    String age = lastFrameNs == 0
                            ? "no frame received"
                            : String.format(Locale.US, "last frame %.1f seconds ago", ageNs / 1e9);
                    report("WAITING: Camera 120 has no fresh frames\n" + age);
                    scheduleCameraRecovery("frame stream stalled");
                }
            }
            mainHandler.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        requestedCameraId = getIntent().getStringExtra(EXTRA_REQUESTED_CAMERA_ID);
        stopAfterFrames = Math.max(0,
                getIntent().getIntExtra(EXTRA_STOP_AFTER_FRAMES, 0));
        audioMode = getIntent().getStringExtra(EXTRA_AUDIO_MODE);
        if (audioMode == null || audioMode.isEmpty()) {
            audioMode = "audiorecord";
        }
        buildUi();
        Log.i(TAG, "Test configuration: requested=" + requestedCameraId
                + "; stopAfterFrames=" + stopAfterFrames);

        cameraThread = new HandlerThread("vcam-camera2");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        mainHandler.post(frameWatchdog);

        if (permissionsGranted()) {
            maybeOpenProcessedCamera();
        } else {
            requestPermissions(new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO
            }, CAMERA_PERMISSION_REQUEST);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        // Target-SDK 35 devices may draw content behind the status bar. Keep
        // the diagnostic heading clear without introducing an AndroidX
        // dependency solely for this small test application.
        root.setPadding(24, 96, 24, 24);
        root.setBackgroundColor(Color.rgb(10, 14, 20));

        TextView title = new TextView(this);
        title.setText("CAMERA2 120 // LIVE TEST");
        title.setTextColor(Color.rgb(126, 231, 135));
        title.setTextSize(20.0f);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        statusView = new TextView(this);
        statusView.setTextSize(15.0f);
        statusView.setTextColor(Color.rgb(215, 224, 234));
        statusView.setPadding(8, 24, 8, 20);
        statusView.setText("Requesting camera permission…");
        root.addView(statusView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        audioStatusView = new TextView(this);
        audioStatusView.setTextSize(14.0f);
        audioStatusView.setTextColor(Color.rgb(255, 191, 105));
        audioStatusView.setPadding(8, 0, 8, 20);
        audioStatusView.setText("MIC: waiting for redirected front-camera session…");
        root.addView(audioStatusView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        AspectFrame previewFrame = new AspectFrame(this);
        previewFrame.setBackgroundColor(Color.BLACK);
        previewView = new SurfaceView(this);
        previewView.getHolder().setFixedSize(WIDTH, HEIGHT);
        previewView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                previewSurface = holder.getSurface();
                surfaceReady = previewSurface != null && previewSurface.isValid();
                maybeOpenProcessedCamera();
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                previewSurface = holder.getSurface();
                surfaceReady = previewSurface != null && previewSurface.isValid();
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                surfaceReady = false;
                previewSurface = null;
                cameraRecoveryScheduled = false;
                mainHandler.removeCallbacks(cameraRecovery);
                closeCamera();
            }
        });
        previewFrame.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        previewCover = new TextView(this);
        previewCover.setText("WAITING FOR CAMERA 120 FRAMES");
        previewCover.setTextColor(Color.rgb(170, 184, 197));
        previewCover.setTextSize(16.0f);
        previewCover.setGravity(Gravity.CENTER);
        previewCover.setBackgroundColor(Color.rgb(2, 4, 6));
        previewFrame.addView(previewCover, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1.0f);
        root.addView(previewFrame, previewParams);

        TextView caption = new TextView(this);
        caption.setText("Live processed stream from external Camera2 device 120");
        caption.setTextColor(Color.rgb(125, 139, 153));
        caption.setTextSize(13.0f);
        caption.setGravity(Gravity.CENTER_HORIZONTAL);
        caption.setPadding(4, 14, 4, 0);
        root.addView(caption, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION_REQUEST
                && permissionsGranted()) {
            maybeOpenProcessedCamera();
        } else {
            report("FAIL: Camera and microphone permissions are required");
            showPreviewCover("CAMERA + MICROPHONE PERMISSIONS REQUIRED");
        }
    }

    private boolean permissionsGranted() {
        return checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void maybeOpenProcessedCamera() {
        if (destroying
                || cameraRecoveryScheduled
                || !surfaceReady
                || cameraOpenPending
                || cameraDevice != null
                || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            String[] ids = manager.getCameraIdList();
            String selected = null;
            if (requestedCameraId != null && !requestedCameraId.isEmpty()) {
                for (String id : ids) {
                    if (requestedCameraId.equals(id)) {
                        selected = id;
                        break;
                    }
                }
                if (selected == null) {
                    report("FAIL: Requested camera " + requestedCameraId
                            + " is not enumerated\nIDs: " + Arrays.toString(ids));
                    showPreviewCover("REQUESTED CAMERA " + requestedCameraId
                            + " IS NOT PUBLISHED");
                    return;
                }
            } else {
                for (String id : ids) {
                    if (PREFERRED_CAMERA_ID.equals(id)) {
                        selected = id;
                        break;
                    }
                }
                if (selected == null) {
                    for (String id : ids) {
                        CameraCharacteristics characteristics =
                                manager.getCameraCharacteristics(id);
                        Integer facing = characteristics.get(
                                CameraCharacteristics.LENS_FACING);
                        if (facing != null
                                && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                            selected = id;
                            break;
                        }
                    }
                }
            }
            if (selected == null) {
                report("FAIL: No external camera found\nIDs: " + Arrays.toString(ids));
                showPreviewCover("CAMERA 120 IS NOT PUBLISHED");
                scheduleCameraRecovery("camera 120 is not published");
                return;
            }

            final String selectedId = selected;
            CameraCharacteristics selectedCharacteristics =
                    manager.getCameraCharacteristics(selectedId);
            Integer facing = selectedCharacteristics.get(
                    CameraCharacteristics.LENS_FACING);
            Integer hardwareLevel = selectedCharacteristics.get(
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
            Integer orientation = selectedCharacteristics.get(
                    CameraCharacteristics.SENSOR_ORIENTATION);
            report("Camera IDs: " + Arrays.toString(ids)
                    + "\nOpening logical camera " + selectedId + "…");
            Log.i(TAG, "Camera IDs=" + Arrays.toString(ids)
                    + "; selected=" + selectedId
                    + "; facing=" + facing
                    + "; hardwareLevel=" + hardwareLevel
                    + "; orientation=" + orientation);
            cameraOpenPending = true;
            final int requestGeneration = ++openGeneration;
            manager.openCamera(selectedId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    if (destroying
                            || requestGeneration != openGeneration
                            || !surfaceReady
                            || cameraDevice != null) {
                        camera.close();
                        return;
                    }
                    cameraOpenPending = false;
                    cameraDevice = camera;
                    Log.i(TAG, "OPENED: requested=" + selectedId
                            + "; actual=" + camera.getId());
                    startAudioVerification();
                    startPreviewAndAnalysis(selectedId, camera.getId());
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    if (requestGeneration == openGeneration) {
                        cameraOpenPending = false;
                        if (cameraDevice == camera) {
                            cameraDevice = null;
                        }
                        report("RECOVERING: Camera " + selectedId + " disconnected");
                        showPreviewCover("CAMERA 120 DISCONNECTED\nRECONNECTING…");
                        scheduleCameraRecovery("camera disconnected");
                    }
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    camera.close();
                    if (requestGeneration == openGeneration) {
                        cameraOpenPending = false;
                        if (cameraDevice == camera) {
                            cameraDevice = null;
                        }
                        report("RECOVERING: Camera " + selectedId + " error=" + error);
                        showPreviewCover("CAMERA OPEN ERROR " + error + "\nRECONNECTING…");
                        scheduleCameraRecovery("camera error " + error);
                    }
                }
            }, cameraHandler);
        } catch (SecurityException | CameraAccessException error) {
            cameraOpenPending = false;
            report("FAIL: Camera enumeration/open failed: " + error);
            showPreviewCover("CAMERA ENUMERATION FAILED");
            Log.e(TAG, "Camera enumeration/open failed", error);
            scheduleCameraRecovery("camera enumeration failed");
        }
    }

    private void scheduleCameraRecovery(final String reason) {
        mainHandler.post(() -> {
            if (destroying || !surfaceReady || cameraRecoveryScheduled) {
                return;
            }
            cameraRecoveryScheduled = true;
            Log.w(TAG, "Scheduling Camera2 recovery: " + reason);
            closeCamera();
            showPreviewCover("RECOVERING CAMERA 120…");
            report("RECOVERING: " + reason + "\nRetrying Camera2 in one second");
            mainHandler.postDelayed(cameraRecovery, RECOVERY_DELAY_MS);
        });
    }

    private void startPreviewAndAnalysis(final String requestedId,
            final String actualId) {
        if (previewSurface == null || !previewSurface.isValid() || cameraDevice == null) {
            report("WAITING: Preview surface is not ready");
            return;
        }
        imageReader = ImageReader.newInstance(WIDTH, HEIGHT, ImageFormat.YUV_420_888, 3);
        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) {
                return;
            }
            try {
                analyzeFrame(requestedId, actualId, image);
            } finally {
                image.close();
            }
        }, cameraHandler);

        Surface analysisSurface = imageReader.getSurface();
        try {
            cameraDevice.createCaptureSession(
                    Arrays.asList(previewSurface, analysisSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                CaptureRequest.Builder request = cameraDevice.createCaptureRequest(
                                        CameraDevice.TEMPLATE_PREVIEW);
                                request.addTarget(previewSurface);
                                request.addTarget(imageReader.getSurface());
                                request.set(CaptureRequest.CONTROL_AE_MODE,
                                        CaptureRequest.CONTROL_AE_MODE_ON);
                                session.setRepeatingRequest(request.build(), null, cameraHandler);
                                report("Requested camera " + requestedId
                                        + " opened as " + actualId
                                        + "; waiting for rendered frames…");
                            } catch (CameraAccessException error) {
                                report("FAIL: Repeating request failed: " + error);
                                showPreviewCover("CAPTURE REQUEST FAILED");
                                Log.e(TAG, "Repeating request failed", error);
                                scheduleCameraRecovery("repeating request failed");
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            report("FAIL: Camera " + requestedId
                                    + " session configuration failed");
                            showPreviewCover("PREVIEW SESSION FAILED");
                            scheduleCameraRecovery("preview session configuration failed");
                        }
                    },
                    cameraHandler);
        } catch (CameraAccessException error) {
            report("FAIL: Capture-session creation failed: " + error);
            showPreviewCover("PREVIEW SESSION FAILED");
            Log.e(TAG, "Capture-session creation failed", error);
            scheduleCameraRecovery("capture-session creation failed");
        }
    }

    private void analyzeFrame(String requestedId, String actualId, Image image) {
        Image.Plane yPlane = image.getPlanes()[0];
        ByteBuffer y = yPlane.getBuffer();
        int rowStride = yPlane.getRowStride();
        int pixelStride = yPlane.getPixelStride();
        long total = 0;
        int minimum = 255;
        int maximum = 0;
        int samples = 0;

        for (int row = 0; row < image.getHeight(); row += LUMA_SAMPLE_STEP) {
            int rowStart = y.position() + row * rowStride;
            for (int column = 0; column < image.getWidth(); column += LUMA_SAMPLE_STEP) {
                int offset = rowStart + column * pixelStride;
                if (offset >= y.limit()) {
                    break;
                }
                int value = y.get(offset) & 0xff;
                total += value;
                minimum = Math.min(minimum, value);
                maximum = Math.max(maximum, value);
                samples++;
            }
        }

        frameCount++;
        long nowNs = System.nanoTime();
        lastFrameNs = nowNs;
        if (firstFrameNs == 0) {
            firstFrameNs = nowNs;
        }
        double elapsedSeconds = Math.max(0.001, (nowNs - firstFrameNs) / 1_000_000_000.0);
        double fps = frameCount > 1 ? (frameCount - 1) / elapsedSeconds : 0.0;
        double mean = samples == 0 ? 0.0 : (double) total / samples;
        String stats = String.format(Locale.US,
                "PASS: Camera2 requested %s, opened %s, and is rendering frames\n"
                        + "Frames: %d   FPS: %.1f\n"
                        + "Frame: %dx%d YUV_420_888\n"
                        + "Luma: min=%d mean=%.2f max=%d\n"
                        + "Sensor timestamp: %d",
                requestedId, actualId, frameCount, fps,
                image.getWidth(), image.getHeight(),
                minimum, mean, maximum, image.getTimestamp());

        if (frameCount == 1) {
            hidePreviewCover();
        }
        if (frameCount == 1 || frameCount % 30 == 0) {
            Log.i(TAG, stats.replace('\n', ' '));
        }
        if (frameCount == 1 || frameCount % 10 == 0) {
            report(stats);
        }
        if (stopAfterFrames > 0
                && frameCount >= stopAfterFrames
                && !stopAfterFramesReached) {
            stopAfterFramesReached = true;
            Log.i(TAG, "AUTO_STOP PASS: requested=" + requestedId
                    + "; actual=" + actualId + "; frames=" + frameCount);
            mainHandler.post(this::finish);
        }
    }

    private void startAudioVerification() {
        stopAudioVerification();
        if (!permissionsGranted()) {
            reportAudio("MIC FAIL: RECORD_AUDIO permission is missing");
            return;
        }
        if ("mediarecorder".equalsIgnoreCase(audioMode)) {
            startMediaRecorderTest();
        } else {
            startAudioRecordTest();
        }
    }

    private void startAudioRecordTest() {
        int minimum = AudioRecord.getMinBufferSize(
                48_000,
                AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT);
        if (minimum <= 0) {
            reportAudio("MIC FAIL: invalid AudioRecord buffer " + minimum);
            return;
        }
        try {
            audioRecord = new AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.CAMCORDER)
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(48_000)
                            .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                            .build())
                    .setBufferSizeInBytes(Math.max(minimum * 2, 19_200))
                    .build();
            audioRecord.startRecording();
            AudioDeviceInfo route = audioRecord.getRoutedDevice();
            reportAudio("MIC: AudioRecord started; route=" + describeRoute(route));
            audioRunning.set(true);
            audioThread = new Thread(() -> runAudioMeter(audioRecord), "vcam-mic-meter");
            audioThread.start();
        } catch (Exception error) {
            reportAudio("MIC FAIL: AudioRecord start: " + error);
            Log.e(TAG, "AudioRecord test failed", error);
            stopAudioVerification();
        }
    }

    private void runAudioMeter(AudioRecord record) {
        short[] pcm = new short[4_800];
        long windows = 0;
        while (audioRunning.get()) {
            int count = record.read(pcm, 0, pcm.length, AudioRecord.READ_BLOCKING);
            if (count <= 0) {
                if (audioRunning.get()) reportAudio("MIC FAIL: read returned " + count);
                break;
            }
            double sumSquares = 0.0;
            int peak = 0;
            int crossings = 0;
            int previous = pcm[0];
            int monoSamples = 0;
            // Inspect the left channel; the relay sends matching stereo PCM.
            for (int index = 0; index + 1 < count; index += 2) {
                int sample = pcm[index];
                sumSquares += (double) sample * sample;
                peak = Math.max(peak, Math.abs(sample));
                if ((previous < 0 && sample >= 0) || (previous >= 0 && sample < 0)) {
                    crossings++;
                }
                previous = sample;
                monoSamples++;
            }
            windows++;
            if (monoSamples > 0 && (windows == 1 || windows % 10 == 0)) {
                double rms = Math.sqrt(sumSquares / monoSamples);
                double dbfs = rms <= 0.0 ? -120.0 : 20.0 * Math.log10(rms / 32768.0);
                double crossingHz = crossings * 48_000.0 / (2.0 * monoSamples);
                AudioDeviceInfo route = record.getRoutedDevice();
                String message = String.format(Locale.US,
                        "MIC PASS: route=%s\nPCM 48k stereo: rms=%.1f (%.1f dBFS) peak=%d zc≈%.0f Hz",
                        describeRoute(route), rms, dbfs, peak, crossingHz);
                reportAudio(message);
                Log.i(TAG, message.replace('\n', ' '));
            }
        }
    }

    private void startMediaRecorderTest() {
        mediaRecorderOutput = new File(getCacheDir(), "virtual-mic-test.m4a");
        try {
            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioSamplingRate(48_000);
            mediaRecorder.setAudioChannels(1);
            mediaRecorder.setAudioEncodingBitRate(128_000);
            mediaRecorder.setOutputFile(mediaRecorderOutput.getAbsolutePath());
            // The Xposed hook selects Remote Submix In immediately before this
            // call, matching Persona's MediaRecorder lifecycle.
            mediaRecorder.prepare();
            AudioDeviceInfo preferred = mediaRecorder.getPreferredDevice();
            mediaRecorder.start();
            reportAudio("MIC: MediaRecorder active; preferred="
                    + describeRoute(preferred) + "\nCapturing five-second AAC proof…");
            mainHandler.postDelayed(stopMediaRecorder, 5_000);
        } catch (Exception error) {
            reportAudio("MIC FAIL: MediaRecorder: " + error);
            Log.e(TAG, "MediaRecorder test failed", error);
            finishMediaRecorderTest();
        }
    }

    private void finishMediaRecorderTest() {
        mainHandler.removeCallbacks(stopMediaRecorder);
        MediaRecorder recorder = mediaRecorder;
        mediaRecorder = null;
        if (recorder == null) return;
        try {
            recorder.stop();
            long size = mediaRecorderOutput == null ? 0 : mediaRecorderOutput.length();
            String message = "MIC PASS: MediaRecorder AAC saved; bytes=" + size
                    + " path=" + mediaRecorderOutput;
            reportAudio(message);
            Log.i(TAG, message);
        } catch (Exception error) {
            reportAudio("MIC FAIL: MediaRecorder stop: " + error);
            Log.e(TAG, "MediaRecorder stop failed", error);
        } finally {
            recorder.release();
        }
    }

    private String describeRoute(AudioDeviceInfo device) {
        if (device == null) return "pending/unknown";
        String kind = device.getType() == AudioDeviceInfo.TYPE_REMOTE_SUBMIX
                ? "REMOTE_SUBMIX" : "type-" + device.getType();
        return kind + "@" + device.getAddress() + "/" + device.getProductName();
    }

    private void reportAudio(final String message) {
        runOnUiThread(() -> audioStatusView.setText(message));
    }

    private void stopAudioVerification() {
        mainHandler.removeCallbacks(stopMediaRecorder);
        audioRunning.set(false);
        AudioRecord record = audioRecord;
        audioRecord = null;
        if (record != null) {
            try { record.stop(); } catch (Exception ignored) {}
            record.release();
        }
        Thread thread = audioThread;
        audioThread = null;
        if (thread != null) {
            try { thread.join(500); } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        if (mediaRecorder != null) {
            finishMediaRecorderTest();
        }
    }

    private void report(final String message) {
        Log.i(TAG, message.replace('\n', ' '));
        runOnUiThread(() -> statusView.setText(message));
    }

    private void showPreviewCover(final String message) {
        runOnUiThread(() -> {
            previewCover.setText(message);
            previewCover.setVisibility(View.VISIBLE);
        });
    }

    private void hidePreviewCover() {
        runOnUiThread(() -> previewCover.setVisibility(View.GONE));
    }

    private void closeCamera() {
        openGeneration++;
        stopAudioVerification();
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
        cameraOpenPending = false;
        frameCount = 0;
        firstFrameNs = 0;
        lastFrameNs = 0;
    }

    @Override
    protected void onDestroy() {
        destroying = true;
        mainHandler.removeCallbacks(frameWatchdog);
        mainHandler.removeCallbacks(cameraRecovery);
        cameraRecoveryScheduled = false;
        closeCamera();
        if (cameraThread != null) {
            cameraThread.quitSafely();
        }
        super.onDestroy();
    }

    private static final class AspectFrame extends FrameLayout {
        AspectFrame(Context context) {
            super(context);
        }

        @Override
        protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
            int width = MeasureSpec.getSize(widthMeasureSpec);
            int desiredHeight = Math.round(width * (HEIGHT / (float) WIDTH));
            int availableHeight = MeasureSpec.getSize(heightMeasureSpec);
            int height = Math.min(desiredHeight, availableHeight);
            super.onMeasure(
                    MeasureSpec.makeMeasureSpec(width, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY));
        }
    }
}
