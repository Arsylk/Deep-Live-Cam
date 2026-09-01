package dev.vcam.bridge;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.RggbChannelVector;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.Log;
import android.util.Range;
import android.view.Surface;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;

/** Camera2 owner and low-latency H.264 producer for the phone bridge. */
public final class CameraBridgeService extends Service {
    public static final String ACTION_CONFIGURE = "dev.vcam.bridge.CONFIGURE";
    public static final String ACTION_START = "dev.vcam.bridge.START";
    public static final String ACTION_STATUS = "dev.vcam.bridge.STATUS";
    public static final String ACTION_STOP = "dev.vcam.bridge.STOP";
    public static final String EXTRA_AE_LOCK = "ae_lock";
    public static final String EXTRA_AWB_LOCK = "awb_lock";
    public static final String EXTRA_EXPOSURE_COMPENSATION = "exposure_compensation";
    public static final String EXTRA_LENS_FACING = "lens_facing";
    public static final String EXTRA_PERSIST = "persist";
    public static final String EXTRA_ROTATION = "rotation";
    public static final String EXTRA_STABILIZATION = "stabilization";
    public static final String EXTRA_STATUS = "status";

    private static final String TAG = "VCamBridge";
    private static final String NOTIFICATION_CHANNEL = "vcam_bridge";
    private static final int NOTIFICATION_ID = 120;
    private static final int TCP_PORT = 10020;
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final int FPS = 30;
    private static final int BITRATE = 10_000_000;
    private static final long INITIAL_CAMERA_RETRY_MS = 2_000;
    private static final long MAX_CAMERA_RETRY_MS = 60_000;

    private final Object socketLock = new Object();
    private final Runnable cameraRetryRunnable = () -> {
        cameraRetryScheduled = false;
        openPhysicalCamera();
    };

    private volatile boolean running;
    private volatile String lensFacing = "front";
    private volatile int exposureCompensation;
    private volatile boolean aeLock;
    private volatile boolean awbLock;
    private volatile String stabilization = "video";
    private volatile String rotation = "auto";
    private volatile int effectiveRotationDegrees;
    private volatile int sensorOrientationDegrees;

    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraManager cameraManager;
    private CameraManager.AvailabilityCallback cameraAvailabilityCallback;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private String selectedPhysicalCameraId;
    private boolean cameraOpenPending;
    private boolean cameraRetryScheduled;
    private long cameraRetryDelayMs = INITIAL_CAMERA_RETRY_MS;

    private MediaCodec encoder;
    private Surface encoderSurface;
    private GlCameraRenderer renderer;
    private Thread drainThread;
    private ServerSocket serverSocket;
    private Thread acceptThread;
    private Socket clientSocket;
    private boolean clientNeedsConfig;
    private byte[] codecConfig;

    private long startedNs;
    private long encodedFrames;
    private long encodedBytes;
    private long capturedFrames;
    private long estimatedDroppedFrames;
    private long lastSensorTimestampNs;
    private double captureIntervalEmaMs;
    private double captureJitterEmaMs;
    private long latestExposureNs;
    private int latestSensitivity;
    private Integer latestAeState;
    private Integer latestAwbState;
    private double exposureJitterEmaEv;
    private double awbGainJitterEma;
    private double lastExposureEv = Double.NaN;
    private double lastRedGainRatio = Double.NaN;
    private double lastBlueGainRatio = Double.NaN;

    private final CameraCaptureSession.CaptureCallback captureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {
                    recordCaptureMetrics(result);
                }
            };

    @Override
    public void onCreate() {
        super.onCreate();
        SharedPreferences preferences = getSharedPreferences("camera", MODE_PRIVATE);
        lensFacing = preferences.getString(EXTRA_LENS_FACING, "front");
        exposureCompensation = preferences.getInt(EXTRA_EXPOSURE_COMPENSATION, 0);
        aeLock = preferences.getBoolean(EXTRA_AE_LOCK, false);
        awbLock = preferences.getBoolean(EXTRA_AWB_LOCK, false);
        stabilization = preferences.getString(EXTRA_STABILIZATION, "video");
        rotation = validatedRotation(preferences.getString(EXTRA_ROTATION, "auto"));
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(
                NOTIFICATION_CHANNEL,
                "Deep Live Camera Bridge",
                NotificationManager.IMPORTANCE_LOW));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        boolean configuring = intent != null && ACTION_CONFIGURE.equals(intent.getAction());
        boolean lensChanged = configuring && readConfiguration(intent);
        startAsForeground();
        if (!running) {
            try {
                startBridge();
            } catch (Exception error) {
                publishStatus("Bridge startup failed: " + error);
                Log.e(TAG, "Bridge startup failed", error);
                stopSelf();
            }
        } else if (configuring && cameraHandler != null) {
            cameraHandler.post(() -> {
                if (lensChanged) {
                    closeActiveCamera();
                    openPhysicalCamera();
                } else {
                    updateRendererRotation();
                    submitCaptureRequest();
                }
            });
        }
        return START_STICKY;
    }

    private boolean readConfiguration(Intent intent) {
        String previousLens = lensFacing;
        if (intent.hasExtra(EXTRA_LENS_FACING)) {
            String requested = intent.getStringExtra(EXTRA_LENS_FACING);
            if ("front".equals(requested) || "back".equals(requested)) {
                lensFacing = requested;
            }
        }
        if (intent.hasExtra(EXTRA_EXPOSURE_COMPENSATION)) {
            exposureCompensation = Math.max(
                    -12,
                    Math.min(12, intent.getIntExtra(EXTRA_EXPOSURE_COMPENSATION, 0)));
        }
        if (intent.hasExtra(EXTRA_AE_LOCK)) {
            aeLock = intent.getBooleanExtra(EXTRA_AE_LOCK, false);
        }
        if (intent.hasExtra(EXTRA_AWB_LOCK)) {
            awbLock = intent.getBooleanExtra(EXTRA_AWB_LOCK, false);
        }
        if (intent.hasExtra(EXTRA_STABILIZATION)) {
            String requested = intent.getStringExtra(EXTRA_STABILIZATION);
            if ("off".equals(requested) || "video".equals(requested) ||
                    "optical".equals(requested)) {
                stabilization = requested;
            }
        }
        if (intent.hasExtra(EXTRA_ROTATION)) {
            rotation = validatedRotation(intent.getStringExtra(EXTRA_ROTATION));
        }
        if (intent.getBooleanExtra(EXTRA_PERSIST, true)) {
            getSharedPreferences("camera", MODE_PRIVATE)
                    .edit()
                    .putString(EXTRA_LENS_FACING, lensFacing)
                    .putInt(EXTRA_EXPOSURE_COMPENSATION, exposureCompensation)
                    .putBoolean(EXTRA_AE_LOCK, aeLock)
                    .putBoolean(EXTRA_AWB_LOCK, awbLock)
                    .putString(EXTRA_STABILIZATION, stabilization)
                    .putString(EXTRA_ROTATION, rotation)
                    .apply();
        }
        return !previousLens.equals(lensFacing);
    }

    private static String validatedRotation(String requested) {
        if ("0".equals(requested) || "90".equals(requested) ||
                "180".equals(requested) || "270".equals(requested)) {
            return requested;
        }
        return "auto";
    }

    private void startAsForeground() {
        Intent activityIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this,
                0,
                activityIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, NOTIFICATION_CHANNEL)
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentTitle("Deep Live Camera Bridge")
                .setContentText("Streaming the physical camera to the Windows processor")
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(
                    NOTIFICATION_ID,
                    notification,
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
    }

    private void startBridge() throws IOException {
        running = true;
        startedNs = System.nanoTime();
        cameraThread = new HandlerThread("vcam-physical-camera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        startTcpServer();
        configureEncoder();
        renderer = new GlCameraRenderer(
                encoderSurface, WIDTH, HEIGHT, WIDTH, HEIGHT);
        renderer.start();
        registerCameraAvailabilityCallback();
        openPhysicalCamera();
        publishStatus(
                "Encoder and GPU rotation renderer ready; waiting for local FFmpeg on " +
                        "127.0.0.1:" + TCP_PORT);
    }

    private void configureEncoder() throws IOException {
        MediaFormat format = MediaFormat.createVideoFormat("video/avc", WIDTH, HEIGHT);
        format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        format.setInteger(MediaFormat.KEY_BIT_RATE, BITRATE);
        format.setInteger(MediaFormat.KEY_BITRATE_MODE,
                MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FPS);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        format.setInteger(MediaFormat.KEY_PROFILE,
                MediaCodecInfo.CodecProfileLevel.AVCProfileHigh);
        format.setInteger(MediaFormat.KEY_LEVEL,
                MediaCodecInfo.CodecProfileLevel.AVCLevel31);
        if (Build.VERSION.SDK_INT >= 29) {
            format.setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0);
        }
        if (Build.VERSION.SDK_INT >= 23) {
            format.setInteger(MediaFormat.KEY_PRIORITY, 0);
            format.setFloat(MediaFormat.KEY_OPERATING_RATE, FPS);
        }
        encoder = MediaCodec.createEncoderByType("video/avc");
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        encoderSurface = encoder.createInputSurface();
        encoder.start();
        drainThread = new Thread(this::drainEncoder, "vcam-h264-drain");
        drainThread.start();
        Log.i(TAG, "Encoder=" + encoder.getName() + " " + WIDTH + "x" + HEIGHT +
                "@" + FPS + " bitrate=" + BITRATE);
    }

    private void startTcpServer() throws IOException {
        serverSocket = new ServerSocket(TCP_PORT, 1, InetAddress.getByName("127.0.0.1"));
        serverSocket.setReuseAddress(true);
        acceptThread = new Thread(this::acceptClients, "vcam-h264-server");
        acceptThread.start();
    }

    private void acceptClients() {
        while (running) {
            try {
                Socket socket = serverSocket.accept();
                socket.setTcpNoDelay(true);
                socket.setSendBufferSize(262_144);
                synchronized (socketLock) {
                    closeClientLocked();
                    clientSocket = socket;
                    clientNeedsConfig = true;
                }
                requestSyncFrame();
                publishStatus("Physical camera streaming to local FFmpeg at " +
                        WIDTH + "x" + HEIGHT + " @ " + FPS + " FPS");
            } catch (IOException error) {
                if (running) {
                    Log.w(TAG, "H.264 client accept failed", error);
                }
            }
        }
    }

    private void registerCameraAvailabilityCallback() {
        cameraManager = getSystemService(CameraManager.class);
        cameraAvailabilityCallback = new CameraManager.AvailabilityCallback() {
            @Override
            public void onCameraAvailable(String cameraId) {
                if (!running || cameraHandler == null) {
                    return;
                }
                try {
                    String selected = selectPhysicalCamera(cameraManager);
                    if (cameraId.equals(selected)) {
                        selectedPhysicalCameraId = selected;
                        if (!cameraRetryScheduled) {
                            cameraHandler.post(CameraBridgeService.this::openPhysicalCamera);
                        }
                    }
                } catch (CameraAccessException error) {
                    scheduleCameraRetry(
                            "Camera availability check failed: " + error.getMessage());
                }
            }
        };
        cameraManager.registerAvailabilityCallback(cameraAvailabilityCallback, cameraHandler);
    }

    private void openPhysicalCamera() {
        if (!running || cameraDevice != null || cameraOpenPending) {
            return;
        }
        if (cameraManager == null) {
            cameraManager = getSystemService(CameraManager.class);
        }
        try {
            String cameraId = selectPhysicalCamera(cameraManager);
            if (cameraId == null) {
                throw new IllegalStateException("no physical camera is available");
            }
            selectedPhysicalCameraId = cameraId;
            CameraCharacteristics characteristics =
                    cameraManager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                Log.i(TAG, "Camera " + cameraId + " texture sizes=" +
                        Arrays.toString(map.getOutputSizes(SurfaceTexture.class)));
            }
            applyRendererRotation(characteristics);
            if (checkSelfPermission(android.Manifest.permission.CAMERA) !=
                    PackageManager.PERMISSION_GRANTED) {
                throw new SecurityException("camera permission was revoked");
            }
            cameraOpenPending = true;
            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice openedCamera) {
                    cameraOpenPending = false;
                    cameraRetryDelayMs = INITIAL_CAMERA_RETRY_MS;
                    cameraRetryScheduled = false;
                    cameraHandler.removeCallbacks(cameraRetryRunnable);
                    if (!running) {
                        openedCamera.close();
                        return;
                    }
                    cameraDevice = openedCamera;
                    createCaptureSession();
                }

                @Override
                public void onDisconnected(CameraDevice disconnectedCamera) {
                    cameraOpenPending = false;
                    disconnectedCamera.close();
                    if (cameraDevice == disconnectedCamera) {
                        cameraDevice = null;
                    }
                    scheduleCameraRetry("Physical camera disconnected");
                }

                @Override
                public void onError(CameraDevice failedCamera, int errorCode) {
                    cameraOpenPending = false;
                    failedCamera.close();
                    if (cameraDevice == failedCamera) {
                        cameraDevice = null;
                    }
                    scheduleCameraRetry("Physical camera error=" + errorCode);
                }
            }, cameraHandler);
        } catch (CameraAccessException | IllegalStateException | SecurityException error) {
            cameraOpenPending = false;
            scheduleCameraRetry("Physical camera open failed: " + error.getMessage());
        }
    }

    private String selectPhysicalCamera(CameraManager manager) throws CameraAccessException {
        int requestedFacing = "back".equals(lensFacing)
                ? CameraCharacteristics.LENS_FACING_BACK
                : CameraCharacteristics.LENS_FACING_FRONT;
        String fallback = null;
        for (String cameraId : manager.getCameraIdList()) {
            Integer facing = manager.getCameraCharacteristics(cameraId).get(
                    CameraCharacteristics.LENS_FACING);
            if (facing != null && facing == CameraCharacteristics.LENS_FACING_EXTERNAL) {
                continue;
            }
            if (facing != null && facing == requestedFacing) {
                return cameraId;
            }
            if (fallback == null) {
                fallback = cameraId;
            }
        }
        return fallback;
    }

    private void createCaptureSession() {
        CameraDevice activeCamera = cameraDevice;
        GlCameraRenderer activeRenderer = renderer;
        if (activeCamera == null || activeRenderer == null) {
            return;
        }
        try {
            activeCamera.createCaptureSession(
                    Collections.singletonList(activeRenderer.getCameraSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            submitCaptureRequest();
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            session.close();
                            closeActiveCamera();
                            scheduleCameraRetry(
                                    "Physical-camera encoder session configuration failed");
                        }
                    },
                    cameraHandler);
        } catch (CameraAccessException | IllegalStateException error) {
            closeActiveCamera();
            scheduleCameraRetry("Capture-session creation failed: " + error.getMessage());
        }
    }

    private void submitCaptureRequest() {
        CameraDevice activeCamera = cameraDevice;
        CameraCaptureSession activeSession = captureSession;
        CameraManager manager = cameraManager;
        String cameraId = selectedPhysicalCameraId;
        GlCameraRenderer activeRenderer = renderer;
        if (activeCamera == null || activeSession == null || manager == null ||
                cameraId == null || activeRenderer == null) {
            return;
        }
        try {
            CaptureRequest.Builder builder = activeCamera.createCaptureRequest(
                    CameraDevice.TEMPLATE_RECORD);
            builder.addTarget(activeRenderer.getCameraSurface());
            builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            if (containsMode(
                    characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES),
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)) {
                builder.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
            }
            Range<Integer> fpsRange = selectFpsRange(characteristics);
            if (fpsRange != null) {
                builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
            }
            Range<Integer> compensationRange = characteristics.get(
                    CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE);
            int appliedExposure = exposureCompensation;
            if (compensationRange != null) {
                appliedExposure = Math.max(
                        compensationRange.getLower(),
                        Math.min(compensationRange.getUpper(), appliedExposure));
                builder.set(
                        CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                        appliedExposure);
            }
            if (Boolean.TRUE.equals(characteristics.get(
                    CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE))) {
                builder.set(CaptureRequest.CONTROL_AE_LOCK, aeLock);
            }
            if (Boolean.TRUE.equals(characteristics.get(
                    CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE))) {
                builder.set(CaptureRequest.CONTROL_AWB_LOCK, awbLock);
            }
            int desiredVideoStabilization = "video".equals(stabilization)
                    ? CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                    : CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF;
            if (containsMode(
                    characteristics.get(
                            CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES),
                    desiredVideoStabilization)) {
                builder.set(
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                        desiredVideoStabilization);
            }
            int desiredOpticalStabilization = "optical".equals(stabilization)
                    ? CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                    : CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF;
            if (containsMode(
                    characteristics.get(
                            CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION),
                    desiredOpticalStabilization)) {
                builder.set(
                        CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                        desiredOpticalStabilization);
            }
            if (containsMode(
                    characteristics.get(
                            CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES),
                    CaptureRequest.NOISE_REDUCTION_MODE_FAST)) {
                builder.set(
                        CaptureRequest.NOISE_REDUCTION_MODE,
                        CaptureRequest.NOISE_REDUCTION_MODE_FAST);
            }
            if (containsMode(
                    characteristics.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES),
                    CaptureRequest.EDGE_MODE_FAST)) {
                builder.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST);
            }
            activeSession.setRepeatingRequest(
                    builder.build(), captureCallback, cameraHandler);
            publishStatus(
                    "Camera " + cameraId + " (" + lensFacing + ") encoding " +
                            WIDTH + "x" + HEIGHT + " @ " + FPS + " FPS; rotation " +
                            rotation + " → " + effectiveRotationDegrees + "° clockwise " +
                            "(aspect-fit), exposure " + appliedExposure + ", AE lock " +
                            (aeLock ? "on" : "off") + ", AWB lock " +
                            (awbLock ? "on" : "off") + ", stabilization " + stabilization);
        } catch (CameraAccessException | IllegalStateException error) {
            closeActiveCamera();
            scheduleCameraRetry("Capture request failed: " + error.getMessage());
        }
    }

    private void updateRendererRotation() {
        CameraManager manager = cameraManager;
        String cameraId = selectedPhysicalCameraId;
        if (manager == null || cameraId == null) {
            return;
        }
        try {
            applyRendererRotation(manager.getCameraCharacteristics(cameraId));
        } catch (CameraAccessException error) {
            Log.w(TAG, "Could not update camera rotation", error);
        }
    }

    private void applyRendererRotation(CameraCharacteristics characteristics) {
        Integer sensor = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        sensorOrientationDegrees = sensor == null ? 0 : sensor;
        if ("auto".equals(rotation)) {
            // Camera2 buffers are in the sensor's landscape orientation.  For
            // a phone in its natural portrait position, back sensors rotate
            // clockwise by SENSOR_ORIENTATION.  Front sensors face the user,
            // so their clockwise correction is the complementary angle.
            effectiveRotationDegrees = facing != null &&
                    facing == CameraCharacteristics.LENS_FACING_FRONT
                    ? (360 - sensorOrientationDegrees) % 360
                    : sensorOrientationDegrees % 360;
        } else {
            effectiveRotationDegrees = Integer.parseInt(rotation);
        }
        if (renderer != null) {
            renderer.setRotationDegrees(effectiveRotationDegrees);
        }
        Log.i(TAG, "Rotation requested=" + rotation + " effective=" +
                effectiveRotationDegrees + " sensor=" + sensorOrientationDegrees +
                " lens=" + String.valueOf(facing));
    }

    private static Range<Integer> selectFpsRange(CameraCharacteristics characteristics) {
        Range<Integer>[] ranges = characteristics.get(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        Range<Integer> selected = null;
        if (ranges == null) {
            return null;
        }
        for (Range<Integer> candidate : ranges) {
            if (candidate.contains(FPS) &&
                    (selected == null || candidate.getLower() > selected.getLower())) {
                selected = candidate;
            }
        }
        return selected;
    }

    private static boolean containsMode(int[] modes, int requested) {
        if (modes == null) {
            return false;
        }
        for (int mode : modes) {
            if (mode == requested) {
                return true;
            }
        }
        return false;
    }

    private void recordCaptureMetrics(TotalCaptureResult result) {
        Long sensorTimestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
        if (sensorTimestamp == null) {
            return;
        }
        capturedFrames++;
        if (lastSensorTimestampNs > 0) {
            double intervalMs = (sensorTimestamp - lastSensorTimestampNs) / 1_000_000.0;
            captureIntervalEmaMs = captureIntervalEmaMs == 0.0
                    ? intervalMs
                    : captureIntervalEmaMs * 0.95 + intervalMs * 0.05;
            captureJitterEmaMs = captureJitterEmaMs * 0.95 +
                    Math.abs(intervalMs - 1000.0 / FPS) * 0.05;
            if (intervalMs > 50.0) {
                estimatedDroppedFrames += Math.max(
                        1L,
                        Math.round(intervalMs / (1000.0 / FPS)) - 1L);
            }
        }
        lastSensorTimestampNs = sensorTimestamp;

        Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
        Integer sensitivity = result.get(CaptureResult.SENSOR_SENSITIVITY);
        if (exposure != null && exposure > 0 && sensitivity != null && sensitivity > 0) {
            latestExposureNs = exposure;
            latestSensitivity = sensitivity;
            double exposureEv = Math.log(exposure.doubleValue() * sensitivity.doubleValue()) /
                    Math.log(2.0);
            if (Double.isFinite(lastExposureEv)) {
                exposureJitterEmaEv = exposureJitterEmaEv * 0.95 +
                        Math.abs(exposureEv - lastExposureEv) * 0.05;
            }
            lastExposureEv = exposureEv;
        }
        RggbChannelVector gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS);
        if (gains != null) {
            double green = Math.max(
                    0.001,
                    (gains.getGreenEven() + gains.getGreenOdd()) * 0.5);
            double redRatio = gains.getRed() / green;
            double blueRatio = gains.getBlue() / green;
            if (Double.isFinite(lastRedGainRatio) && Double.isFinite(lastBlueGainRatio)) {
                awbGainJitterEma = awbGainJitterEma * 0.95 +
                        Math.hypot(
                                redRatio - lastRedGainRatio,
                                blueRatio - lastBlueGainRatio) * 0.05;
            }
            lastRedGainRatio = redRatio;
            lastBlueGainRatio = blueRatio;
        }
        latestAeState = result.get(CaptureResult.CONTROL_AE_STATE);
        latestAwbState = result.get(CaptureResult.CONTROL_AWB_STATE);
    }

    private void drainEncoder() {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (running) {
            try {
                int index = encoder.dequeueOutputBuffer(info, 10_000);
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    cacheCodecConfig(encoder.getOutputFormat());
                    continue;
                }
                if (index < 0) {
                    continue;
                }
                ByteBuffer source = encoder.getOutputBuffer(index);
                if (source != null && info.size > 0) {
                    ByteBuffer data = source.duplicate();
                    data.position(info.offset);
                    data.limit(info.offset + info.size);
                    byte[] accessUnit = new byte[info.size];
                    data.get(accessUnit);
                    byte[] annexB = toAnnexB(accessUnit);
                    if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                        codecConfig = annexB;
                    } else {
                        sendAccessUnit(annexB);
                        encodedFrames++;
                        encodedBytes += annexB.length;
                        if (encodedFrames % 300 == 0) {
                            logMetrics();
                        }
                    }
                }
                encoder.releaseOutputBuffer(index, false);
            } catch (IllegalStateException error) {
                if (running) {
                    Log.e(TAG, "Encoder drain failed", error);
                    publishStatus("Encoder stopped unexpectedly: " + error.getMessage());
                }
                return;
            }
        }
    }

    private void logMetrics() {
        double seconds = (System.nanoTime() - startedNs) / 1_000_000_000.0;
        double encodedFps = encodedFrames / Math.max(0.001, seconds);
        double bitrateMbps = (encodedBytes * 8.0 / Math.max(0.001, seconds)) / 1_000_000.0;
        Log.i(TAG, String.format(
                Locale.US,
                "encoded=%d fps=%.1f bitrate=%.2fMbps " +
                        "captureInterval=%.2fms jitter=%.2fms dropped~%d " +
                        "exposure=%.3fms ISO=%d exposureJitter=%.4fEV " +
                        "awbJitter=%.5f ae=%s awb=%s rotation=%s effectiveRotation=%d " +
                        "rendered=%d textureRotation=%d shaderRotation=%d",
                encodedFrames,
                encodedFps,
                bitrateMbps,
                captureIntervalEmaMs,
                captureJitterEmaMs,
                estimatedDroppedFrames,
                latestExposureNs / 1_000_000.0,
                latestSensitivity,
                exposureJitterEmaEv,
                awbGainJitterEma,
                String.valueOf(latestAeState),
                String.valueOf(latestAwbState),
                rotation,
                effectiveRotationDegrees,
                renderer == null ? 0 : renderer.getRenderedFrames(),
                renderer == null ? 0 : renderer.getTextureMatrixRotationDegrees(),
                renderer == null ? 0 : renderer.getShaderRotationDegrees()));
    }

    private void cacheCodecConfig(MediaFormat format) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        for (String key : Arrays.asList("csd-0", "csd-1")) {
            ByteBuffer source = format.getByteBuffer(key);
            if (source == null) {
                continue;
            }
            ByteBuffer copy = source.duplicate();
            byte[] data = new byte[copy.remaining()];
            copy.get(data);
            byte[] annexB = toAnnexB(data);
            output.write(annexB, 0, annexB.length);
        }
        if (output.size() > 0) {
            codecConfig = output.toByteArray();
            Log.i(TAG, "Cached " + codecConfig.length + " bytes of AVC configuration");
        }
    }

    private static byte[] toAnnexB(byte[] source) {
        if (source.length >= 4 && source[0] == 0 && source[1] == 0 &&
                (source[2] == 1 || (source[2] == 0 && source[3] == 1))) {
            return source;
        }
        ByteArrayOutputStream output = new ByteArrayOutputStream(source.length + 16);
        int offset = 0;
        while (offset + 4 <= source.length) {
            int size = ((source[offset] & 0xff) << 24) |
                    ((source[offset + 1] & 0xff) << 16) |
                    ((source[offset + 2] & 0xff) << 8) |
                    (source[offset + 3] & 0xff);
            int start = offset + 4;
            if (size <= 0 || start + size > source.length) {
                output.reset();
                break;
            }
            output.write(0);
            output.write(0);
            output.write(0);
            output.write(1);
            output.write(source, start, size);
            offset = start + size;
        }
        if (offset == source.length && output.size() > 0) {
            return output.toByteArray();
        }
        byte[] prefixed = new byte[source.length + 4];
        prefixed[3] = 1;
        System.arraycopy(source, 0, prefixed, 4, source.length);
        return prefixed;
    }

    private void sendAccessUnit(byte[] accessUnit) {
        synchronized (socketLock) {
            if (clientSocket == null) {
                return;
            }
            try {
                OutputStream output = clientSocket.getOutputStream();
                if (clientNeedsConfig && codecConfig != null) {
                    output.write(codecConfig);
                    clientNeedsConfig = false;
                }
                output.write(accessUnit);
            } catch (IOException error) {
                Log.w(TAG, "Local FFmpeg disconnected");
                closeClientLocked();
            }
        }
    }

    private void requestSyncFrame() {
        MediaCodec activeEncoder = encoder;
        if (activeEncoder == null) {
            return;
        }
        try {
            Bundle parameters = new Bundle();
            parameters.putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0);
            activeEncoder.setParameters(parameters);
        } catch (IllegalStateException error) {
            Log.w(TAG, "Could not request an IDR frame", error);
        }
    }

    private void scheduleCameraRetry(String reason) {
        Log.w(TAG, reason);
        if (!running || cameraHandler == null || cameraRetryScheduled) {
            return;
        }
        long delay = cameraRetryDelayMs;
        cameraRetryDelayMs = Math.min(cameraRetryDelayMs * 2, MAX_CAMERA_RETRY_MS);
        cameraRetryScheduled = true;
        publishStatus(reason + "; waiting for availability (fallback retry in " +
                delay / 1000 + " seconds)");
        cameraHandler.postDelayed(cameraRetryRunnable, delay);
    }

    private void closeActiveCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        lastSensorTimestampNs = 0;
        lastExposureEv = Double.NaN;
        lastRedGainRatio = Double.NaN;
        lastBlueGainRatio = Double.NaN;
    }

    private void publishStatus(String status) {
        Log.i(TAG, status);
        Intent broadcast = new Intent(ACTION_STATUS);
        broadcast.setPackage(getPackageName());
        broadcast.putExtra(EXTRA_STATUS, status);
        sendBroadcast(broadcast);
    }

    @Override
    public void onDestroy() {
        running = false;
        cameraOpenPending = false;
        cameraRetryScheduled = false;
        synchronized (socketLock) {
            closeClientLocked();
        }
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
            serverSocket = null;
        }
        if (cameraHandler != null) {
            cameraHandler.removeCallbacks(cameraRetryRunnable);
        }
        if (cameraManager != null && cameraAvailabilityCallback != null) {
            cameraManager.unregisterAvailabilityCallback(cameraAvailabilityCallback);
            cameraAvailabilityCallback = null;
        }
        closeActiveCamera();
        if (renderer != null) {
            renderer.close();
            renderer = null;
        }
        if (encoder != null) {
            try {
                encoder.stop();
            } catch (IllegalStateException ignored) {
            }
            encoder.release();
            encoder = null;
        }
        if (encoderSurface != null) {
            encoderSurface.release();
            encoderSurface = null;
        }
        if (cameraThread != null) {
            cameraThread.quitSafely();
            cameraThread = null;
            cameraHandler = null;
        }
        publishStatus("Bridge stopped; processed camera remains on fallback");
        super.onDestroy();
    }

    private void closeClientLocked() {
        if (clientSocket != null) {
            try {
                clientSocket.close();
            } catch (IOException ignored) {
            }
            clientSocket = null;
        }
        clientNeedsConfig = false;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
