package dev.vcam.mobile;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.opencv.core.Mat;

final class CameraController implements AutoCloseable {
    interface Listener {
        void onCameraFrame(Mat bgr, long timestampNs, long generation);
        void onCameraState(String state);
        void onCameraMetrics(double captureFps, long dropped);
    }

    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private final Context context;
    private final Listener listener;
    private final CameraManager manager;
    private HandlerThread thread;
    private Handler handler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private ImageReader reader;
    private volatile boolean front = true;
    private volatile long generation;
    private long frameCount, dropped;
    private long metricsStarted;
    private boolean fixedThirtySupported;
    private final java.util.concurrent.atomic.AtomicBoolean busy = new java.util.concurrent.atomic.AtomicBoolean();
    // Camera callbacks can arrive after a lens switch. A monotonically
    // increasing attempt token prevents an obsolete callback from closing a
    // newer camera or ImageReader.
    private long nextOpenAttempt;
    private long activeOpenAttempt = -1;
    private final Set<String> unavailableCameraIds = new HashSet<>();
    private volatile boolean closing;
    private final CameraManager.AvailabilityCallback availability =
            new CameraManager.AvailabilityCallback() {
                @Override public void onCameraAvailable(String cameraId) {
                    unavailableCameraIds.remove(cameraId);
                    Handler active = handler;
                    if (!closing && active != null && camera == null && !"120".equals(cameraId)
                            && !anotherPhysicalCameraIsBusy()) {
                        active.removeCallbacks(CameraController.this::openSelected);
                        active.postDelayed(CameraController.this::openSelected, 250);
                    }
                }

                @Override public void onCameraUnavailable(String cameraId) {
                    unavailableCameraIds.add(cameraId);
                }
            };

    CameraController(Context context, Listener listener) {
        this.context = context;
        this.listener = listener;
        manager = context.getSystemService(CameraManager.class);
    }

    boolean isFront() { return front; }
    long generation() { return generation; }

    void start() {
        if (thread != null) return;
        closing = false;
        thread = new HandlerThread("mobile-camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        manager.registerAvailabilityCallback(availability, handler);
        // Registration first reports the availability of every camera. Give
        // those callbacks time to establish the preflight view before asking
        // Camera2 to arbitrate ownership; a foreground open could otherwise
        // evict a bridge or camera app that is already using another lens.
        handler.postDelayed(this::openSelected, 500);
    }

    void switchLens() {
        front = !front;
        generation++;
        Handler active = handler;
        if (active == null || closing) return;
        active.post(() -> {
            closeCameraOnly();
            openSelected();
        });
    }

    private void openSelected() {
        if (closing || camera != null || activeOpenAttempt >= 0) return;
        final long attempt = ++nextOpenAttempt;
        final long requestedGeneration = generation;
        activeOpenAttempt = attempt;
        try {
            if (context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                listener.onCameraState("Camera permission required");
                finishAttempt(attempt);
                return;
            }
            if (anotherPhysicalCameraIsBusy()) {
                listener.onCameraState("Another camera stack is active — waiting without interrupting it");
                finishAttempt(attempt);
                if (!closing && handler != null) handler.postDelayed(this::openSelected, 1500);
                return;
            }
            String cameraId = selectCamera();
            if (cameraId == null) {
                listener.onCameraState("No " + (front ? "front" : "rear") + " physical camera available");
                finishAttempt(attempt);
                if (!closing && handler != null) handler.postDelayed(this::openSelected, 1500);
                return;
            }
            CameraCharacteristics c = manager.getCameraCharacteristics(cameraId);
            Integer orientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Range<Integer>[] ranges = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
            fixedThirtySupported = false;
            if (ranges != null) for (Range<Integer> range : ranges) {
                if (range.getLower() == 30 && range.getUpper() == 30) fixedThirtySupported = true;
            }
            final ImageReader openedReader = ImageReader.newInstance(
                    WIDTH, HEIGHT, ImageFormat.YUV_420_888, 2);
            reader = openedReader;
            int sensorOrientation = orientation == null ? 0 : orientation;
            openedReader.setOnImageAvailableListener(r -> onImage(r, sensorOrientation), handler);
            listener.onCameraState("Opening physical camera " + cameraId + "…");
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice device) {
                    if (closing || activeOpenAttempt != attempt
                            || generation != requestedGeneration || reader != openedReader) {
                        device.close(); openedReader.close();
                        if (reader == openedReader) reader = null;
                        return;
                    }
                    finishAttempt(attempt);
                    camera = device;
                    createSession(device, openedReader, requestedGeneration);
                }
                @Override public void onDisconnected(CameraDevice device) {
                    handleDeviceLoss(device, openedReader, attempt, requestedGeneration,
                            "Physical camera disconnected", true, 1000);
                }
                @Override public void onError(CameraDevice device, int error) {
                    boolean busyError = error == ERROR_CAMERA_IN_USE || error == ERROR_MAX_CAMERAS_IN_USE;
                    handleDeviceLoss(device, openedReader, attempt, requestedGeneration,
                            busyError ? "Camera is in use by another app; preview will resume when available"
                                    : "Physical camera error " + error,
                            !busyError, 1500);
                }
            }, handler);
        } catch (Exception error) {
            if (activeOpenAttempt != attempt) return;
            finishAttempt(attempt);
            closeCameraOnly();
            listener.onCameraState("Camera unavailable: " + error.getMessage());
            if (!closing && handler != null) handler.postDelayed(this::openSelected, 1500);
        }
    }

    private void finishAttempt(long attempt) {
        if (activeOpenAttempt == attempt) activeOpenAttempt = -1;
    }

    private void handleDeviceLoss(
            CameraDevice device, ImageReader expectedReader, long attempt,
            long expectedGeneration, String state, boolean retry, long delayMs) {
        boolean ownsCurrent = (camera == device || (camera == null && reader == expectedReader))
                && generation == expectedGeneration;
        finishAttempt(attempt);
        device.close();
        if (!ownsCurrent) {
            expectedReader.close();
            return;
        }
        closeCameraOnly();
        listener.onCameraState(state);
        if (retry && !closing && handler != null) {
            handler.postDelayed(this::openSelected, delayMs);
        }
    }

    private String selectCamera() throws CameraAccessException {
        int facing = front ? CameraCharacteristics.LENS_FACING_FRONT
                : CameraCharacteristics.LENS_FACING_BACK;
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            Integer value = c.get(CameraCharacteristics.LENS_FACING);
            if (value != null && value == CameraCharacteristics.LENS_FACING_EXTERNAL) continue;
            // Never admit the virtual camera into its own input pipeline.
            if ("120".equals(id)) continue;
            if (value != null && value == facing) return id;
        }
        return null;
    }

    private boolean anotherPhysicalCameraIsBusy() {
        for (String id : unavailableCameraIds) {
            if ("120".equals(id)) continue;
            try {
                Integer facing = manager.getCameraCharacteristics(id).get(
                        CameraCharacteristics.LENS_FACING);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_EXTERNAL) return true;
            } catch (CameraAccessException ignored) {
                // An ID that disappeared while unavailable is conservatively
                // considered busy until CameraManager reports it available.
                return true;
            }
        }
        return false;
    }

    private void createSession(
            CameraDevice device, ImageReader activeReader, long expectedGeneration) {
        if (camera != device || reader != activeReader || generation != expectedGeneration) return;
        try {
            device.createCaptureSession(Arrays.asList(activeReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override public void onConfigured(CameraCaptureSession created) {
                            if (closing || camera != device || reader != activeReader
                                    || generation != expectedGeneration) {
                                created.close(); return;
                            }
                            session = created;
                            try {
                                CaptureRequest.Builder request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                request.addTarget(activeReader.getSurface());
                                request.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                                if (fixedThirtySupported) request.set(
                                        CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
                                created.setRepeatingRequest(request.build(), null, handler);
                                metricsStarted = System.nanoTime(); frameCount = 0;
                                listener.onCameraState("Live " + (front ? "front" : "rear") + " camera");
                            } catch (Exception error) {
                                failSession(created, device, activeReader, expectedGeneration,
                                        "Capture request failed: " + error.getMessage());
                            }
                        }
                        @Override public void onConfigureFailed(CameraCaptureSession failed) {
                            failSession(failed, device, activeReader, expectedGeneration,
                                    "Camera stream configuration failed");
                        }
                    }, handler);
        } catch (CameraAccessException error) {
            failSession(null, device, activeReader, expectedGeneration,
                    "Camera session failed: " + error.getMessage());
        }
    }

    private void failSession(
            CameraCaptureSession failed, CameraDevice device, ImageReader expectedReader,
            long expectedGeneration, String state) {
        if (failed != null) failed.close();
        if (camera != device || reader != expectedReader || generation != expectedGeneration) return;
        closeCameraOnly();
        listener.onCameraState(state);
        if (!closing && handler != null) handler.postDelayed(this::openSelected, 1000);
    }

    private void onImage(ImageReader source, int sensorOrientation) {
        Image image = null;
        try {
            image = source.acquireLatestImage();
            if (image == null) return;
            if (!busy.compareAndSet(false, true)) { dropped++; return; }
            Mat sensor = FrameTransforms.imageToBgr(image);
            Mat display = FrameTransforms.rotateToDisplay(sensor, sensorOrientation, front);
            sensor.release();
            frameCount++;
            listener.onCameraFrame(display, image.getTimestamp(), generation);
            display.release();
            long elapsed = System.nanoTime() - metricsStarted;
            if (elapsed >= 1_000_000_000L) {
                listener.onCameraMetrics(frameCount * 1_000_000_000.0 / elapsed, dropped);
                frameCount = 0; metricsStarted = System.nanoTime();
            }
        } catch (Throwable error) {
            listener.onCameraState("Camera frame error: " + error.getMessage());
        } finally {
            busy.set(false);
            if (image != null) image.close();
        }
    }

    private void closeCameraOnly() {
        activeOpenAttempt = -1;
        generation++;
        CameraCaptureSession oldSession = session; session = null;
        CameraDevice oldCamera = camera; camera = null;
        ImageReader oldReader = reader; reader = null;
        if (oldSession != null) {
            try { oldSession.stopRepeating(); } catch (Exception ignored) {}
            oldSession.close();
        }
        if (oldCamera != null) oldCamera.close();
        if (oldReader != null) oldReader.close();
    }

    @Override public void close() {
        closing = true;
        try { manager.unregisterAvailabilityCallback(availability); } catch (Exception ignored) {}
        CountDownLatch closed = new CountDownLatch(1);
        Handler active = handler;
        if (active != null) {
            active.removeCallbacksAndMessages(null);
            active.post(() -> { closeCameraOnly(); closed.countDown(); });
            try { closed.await(2, TimeUnit.SECONDS); } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        if (thread != null) {
            thread.quitSafely();
            try { thread.join(2000); } catch (InterruptedException ignored) {}
        }
        thread = null; handler = null;
    }
}
