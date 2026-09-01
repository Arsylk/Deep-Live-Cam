package dev.vcam.mobile;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

final class FacePipeline implements AutoCloseable {
    interface Listener {
        void onProcessed(Mat bgr, long timestampNs, long cameraGeneration, long sourceGeneration);
        void onMetrics(Metrics metrics);
        void onError(String detail);
    }

    static final class Metrics {
        final String provider;
        final double detectionMs, swapMs, totalMs, inferenceFps;
        final long frames, dropped, sourceAgeMs;
        final boolean faceFound, sourceReady;

        Metrics(String provider, double detectionMs, double swapMs, double totalMs,
                double inferenceFps, long frames, long dropped, long sourceAgeMs,
                boolean faceFound, boolean sourceReady) {
            this.provider = provider; this.detectionMs = detectionMs; this.swapMs = swapMs;
            this.totalMs = totalMs; this.inferenceFps = inferenceFps; this.frames = frames;
            this.dropped = dropped; this.sourceAgeMs = sourceAgeMs;
            this.faceFound = faceFound; this.sourceReady = sourceReady;
        }
    }

    private final InferenceRuntime runtime;
    private final ModelStore models;
    private final Listener listener;
    private final Mat softMask = FrameTransforms.softMask(128);
    private final AtomicBoolean running = new AtomicBoolean(true);
    private final AtomicBoolean resourcesReleased = new AtomicBoolean();
    private final AtomicLong sourceGeneration = new AtomicLong();
    private final Object frameLock = new Object();
    private final Thread worker;
    private Frame pending;
    private volatile float[] latent;
    private volatile long sourceSetAt;
    private long processed, dropped;
    private double detectionEma, swapEma, totalEma, fpsEma;
    private long previousEndNs;
    private boolean faceFound;
    private Face trackedFace;
    private int trackingMisses;
    private long lastTrackingTimestampNs;
    private long trackingCameraGeneration = -1, trackingSourceGeneration = -1;
    private final double[] colorScale = {Double.NaN, Double.NaN, Double.NaN};
    private final double[] colorOffset = {Double.NaN, Double.NaN, Double.NaN};

    private static final class Frame {
        final Mat bgr;
        final long timestamp, cameraGeneration;
        Frame(Mat bgr, long timestamp, long generation) {
            this.bgr = bgr; this.timestamp = timestamp; this.cameraGeneration = generation;
        }
    }

    FacePipeline(ModelStore models, Listener listener) throws Exception {
        this.models = models;
        this.listener = listener;
        runtime = new InferenceRuntime(models);
        worker = new Thread(this::loop, "mobile-face-pipeline");
        worker.start();
    }

    long sourceGeneration() { return sourceGeneration.get(); }
    boolean sourceReady() { return latent != null; }
    String provider() { return runtime.provider(); }

    void offer(Mat bgr, long timestampNs, long cameraGeneration) {
        Mat owned = bgr.clone();
        synchronized (frameLock) {
            if (pending != null) {
                pending.bgr.release();
                dropped++;
            }
            pending = new Frame(owned, timestampNs, cameraGeneration);
            frameLock.notifyAll();
        }
    }

    synchronized float[] setSource(Mat sourceBgr) throws Exception {
        if (!running.get()) throw new IllegalStateException("Pipeline is closed");
        Face face = bestFace(sourceBgr, true);
        if (face == null) throw new IllegalArgumentException("No clear face found in source image");
        Mat matrix = new Mat();
        Mat aligned = FrameTransforms.align(sourceBgr, face.keypoints, 112, matrix);
        matrix.release();
        if (aligned.empty()) throw new IllegalArgumentException("Could not align source face");
        float[] embedding;
        try {
            embedding = runtime.recognize(FrameTransforms.nchwRgb(aligned, 127.5f, 127.5f));
        } finally {
            aligned.release();
        }
        normalize(embedding);
        float[] emap = readEmap();
        float[] mapped = new float[512];
        for (int col = 0; col < 512; col++) {
            double value = 0;
            for (int row = 0; row < 512; row++) value += embedding[row] * emap[row * 512 + col];
            mapped[col] = (float) value;
        }
        normalize(mapped);
        if (!running.get()) throw new IllegalStateException("Pipeline closed while loading source");
        setSourceLatent(mapped);
        return mapped.clone();
    }

    synchronized void setSourceLatent(float[] mapped) {
        if (!running.get()) throw new IllegalStateException("Pipeline is closed");
        if (mapped == null || mapped.length != 512) {
            throw new IllegalArgumentException("Source embedding must contain 512 floats");
        }
        latent = mapped.clone();
        sourceSetAt = android.os.SystemClock.elapsedRealtime();
        sourceGeneration.incrementAndGet();
    }

    private float[] readEmap() throws Exception {
        byte[] bytes;
        try (FileInputStream input = new FileInputStream(models.file(ModelStore.EMAP))) {
            bytes = ModelStore.readAll(input);
        }
        FloatBuffer floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] values = new float[512 * 512];
        floats.get(values);
        return values;
    }

    private void loop() {
        try {
        while (running.get()) {
            Frame frame;
            synchronized (frameLock) {
                while (running.get() && pending == null) {
                    try { frameLock.wait(); } catch (InterruptedException ignored) {}
                }
                frame = pending; pending = null;
            }
            if (frame == null) continue;
            long started = System.nanoTime();
            long generation = sourceGeneration.get();
            Mat output = null;
            boolean transferred = false;
            try {
                float[] activeLatent = latent;
                output = frame.bgr.clone();
                if (activeLatent != null) {
                    long detectStarted = System.nanoTime();
                    Face face = bestFace(frame.bgr, false);
                    detectionEma = ema(detectionEma, elapsedMs(detectStarted));
                    if (trackingCameraGeneration != frame.cameraGeneration
                            || trackingSourceGeneration != generation) resetTracking(frame.cameraGeneration, generation);
                    Face tracked = updateTracking(face, frame.timestamp);
                    faceFound = face != null;
                    if (tracked != null) {
                        Mat affine = new Mat(), aligned = null, generated = null;
                        Mat colorTarget = null, matched = null, pasted = null;
                        try {
                            aligned = FrameTransforms.align(frame.bgr, tracked.keypoints, 128, affine);
                            if (aligned.empty()) continue;
                            long swapStarted = System.nanoTime();
                            float[] prediction = runtime.swap(
                                    FrameTransforms.nchwRgb(aligned, 0.0f, 255.0f), activeLatent);
                            generated = FrameTransforms.rgbOutputToBgr(prediction, 128);
                            colorTarget = FrameTransforms.alignedWithBorder(
                                    frame.bgr, affine, 128, Core.BORDER_REFLECT_101);
                            matched = FrameTransforms.colorMatch(
                                    generated, colorTarget, softMask, .35f, colorScale, colorOffset);
                            pasted = FrameTransforms.paste(frame.bgr, matched, affine, softMask);
                            long missingNs = Math.max(0, frame.timestamp - lastTrackingTimestampNs);
                            float alpha = face != null ? 1.0f
                                    : Math.max(0, 1.0f - missingNs / 2_500_000_000.0f);
                            if (alpha < .999f) Core.addWeighted(frame.bgr, 1.0 - alpha, pasted, alpha, 0, pasted);
                            FrameTransforms.sharpenFace(pasted, tracked.bbox, .2f);
                            output.release(); output = pasted;
                            pasted = null;
                            swapEma = ema(swapEma, elapsedMs(swapStarted));
                        } finally {
                            affine.release();
                            if (aligned != null) aligned.release();
                            if (generated != null) generated.release();
                            if (colorTarget != null) colorTarget.release();
                            if (matched != null) matched.release();
                            if (pasted != null) pasted.release();
                        }
                    }
                } else faceFound = false;
                if (generation == sourceGeneration.get()) {
                    processed++;
                    totalEma = ema(totalEma, elapsedMs(started));
                    long now = System.nanoTime();
                    if (previousEndNs != 0) fpsEma = ema(fpsEma, 1_000_000_000.0 / (now - previousEndNs));
                    previousEndNs = now;
                    listener.onProcessed(output, frame.timestamp, frame.cameraGeneration, generation);
                    transferred = true;
                    listener.onMetrics(snapshot());
                }
            } catch (Throwable error) {
                listener.onError(error.getClass().getSimpleName() + ": " + error.getMessage());
                // A provider failure must not leave the last processed frame
                // frozen on screen. Hand the current raw clone to the UI, then
                // MainActivity switches processing off for subsequent frames.
                if (output != null) {
                    listener.onProcessed(output, frame.timestamp, frame.cameraGeneration, generation);
                    transferred = true;
                }
            } finally {
                if (!transferred && output != null) output.release();
                frame.bgr.release();
            }
        }
        } finally {
            releaseResources();
        }
    }

    private Face bestFace(Mat original, boolean leftmost) throws Exception {
        int ow = original.cols(), oh = original.rows();
        float ratio = (float) oh / ow;
        int rw, rh;
        if (ratio > 1.0f) { rh = 640; rw = Math.max(1, (int) (640 / ratio)); }
        else { rw = 640; rh = Math.max(1, (int) (640 * ratio)); }
        float scale = (float) rh / oh;
        Mat resized = new Mat();
        Imgproc.resize(original, resized, new Size(rw, rh));
        Mat canvas = Mat.zeros(640, 640, original.type());
        resized.copyTo(canvas.submat(0, rh, 0, rw));
        float[][] values;
        try {
            values = runtime.detect(FrameTransforms.nchwRgb(canvas, 127.5f, 128.0f));
        } finally {
            resized.release(); canvas.release();
        }
        int[] strides = {8, 16, 32};
        List<Face> candidates = new ArrayList<>();
        for (int level = 0; level < 3; level++) {
            float[] scores = values[level];
            float[] boxes = values[3 + level];
            float[] points = values[6 + level];
            int stride = strides[level];
            int width = 640 / stride;
            for (int i = 0; i < scores.length; i++) {
                float score = scores[i];
                if (score < 0.5f) continue;
                int cell = i / 2;
                float cx = (cell % width) * stride;
                float cy = (cell / width) * stride;
                float[] bbox = {
                        (cx - boxes[i * 4] * stride) / scale,
                        (cy - boxes[i * 4 + 1] * stride) / scale,
                        (cx + boxes[i * 4 + 2] * stride) / scale,
                        (cy + boxes[i * 4 + 3] * stride) / scale,
                };
                float[] kps = new float[10];
                for (int point = 0; point < 5; point++) {
                    kps[point * 2] = (cx + points[i * 10 + point * 2] * stride) / scale;
                    kps[point * 2 + 1] = (cy + points[i * 10 + point * 2 + 1] * stride) / scale;
                }
                candidates.add(new Face(bbox, kps, score));
            }
        }
        candidates.sort(Comparator.comparingDouble((Face f) -> -f.score));
        List<Face> kept = new ArrayList<>();
        for (Face candidate : candidates) {
            boolean suppressed = false;
            for (Face existing : kept) if (iou(candidate, existing) > 0.4f) { suppressed = true; break; }
            if (!suppressed) kept.add(candidate);
        }
        if (!leftmost) kept.removeIf(face -> face.width() < 64 || face.height() < 64);
        if (kept.isEmpty()) return null;
        if (leftmost) return kept.stream().min(Comparator.comparingDouble(f -> f.bbox[0])).orElse(null);
        if (trackedFace != null) {
            Face associated = kept.stream().max(Comparator.comparingDouble(f -> iou(f, trackedFace))).orElse(null);
            if (associated != null && iou(associated, trackedFace) >= .10f) return associated;
        }
        return kept.stream().max(Comparator.comparingDouble(f -> f.score * Math.sqrt(f.area()))).orElse(null);
    }

    private static float iou(Face a, Face b) {
        float x1 = Math.max(a.bbox[0], b.bbox[0]), y1 = Math.max(a.bbox[1], b.bbox[1]);
        float x2 = Math.min(a.bbox[2], b.bbox[2]), y2 = Math.min(a.bbox[3], b.bbox[3]);
        float intersection = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        return intersection / Math.max(1, a.area() + b.area() - intersection);
    }

    private Face updateTracking(Face detection, long timestampNs) {
        if (detection != null) {
            if (trackedFace == null) trackedFace = detection;
            else {
                // Keep the original ~0.28 new-sample weight at 30 FPS, but
                // converge rapidly when this large model only produces one
                // inference every several hundred milliseconds.
                double elapsed = lastTrackingTimestampNs == 0 ? 1.0
                        : Math.max(0, (timestampNs - lastTrackingTimestampNs) / 1_000_000_000.0);
                float fresh = (float) Math.max(.28,
                        Math.min(1.0, 1.0 - Math.exp(-elapsed / .10)));
                float[] bbox = new float[4], points = new float[10];
                for (int i = 0; i < 4; i++) bbox[i] = trackedFace.bbox[i] * (1 - fresh) + detection.bbox[i] * fresh;
                for (int i = 0; i < 10; i++) points[i] = trackedFace.keypoints[i] * (1 - fresh) + detection.keypoints[i] * fresh;
                trackedFace = new Face(bbox, points, detection.score);
            }
            trackingMisses = 0; lastTrackingTimestampNs = timestampNs;
        } else if (trackedFace != null) {
            trackingMisses++;
            if (timestampNs - lastTrackingTimestampNs > 2_500_000_000L) trackedFace = null;
        }
        return trackedFace;
    }

    private void resetTracking(long cameraGeneration, long sourceGeneration) {
        trackedFace = null; trackingMisses = 0; lastTrackingTimestampNs = 0;
        trackingCameraGeneration = cameraGeneration; trackingSourceGeneration = sourceGeneration;
        java.util.Arrays.fill(colorScale, Double.NaN);
        java.util.Arrays.fill(colorOffset, Double.NaN);
    }

    private Metrics snapshot() {
        return new Metrics(runtime.provider(), detectionEma, swapEma, totalEma, fpsEma,
                processed, dropped, sourceSetAt == 0 ? -1 : android.os.SystemClock.elapsedRealtime() - sourceSetAt,
                faceFound, latent != null);
    }

    private static double elapsedMs(long start) { return (System.nanoTime() - start) / 1_000_000.0; }
    private static double ema(double previous, double value) { return previous == 0 ? value : previous * .85 + value * .15; }
    private static void normalize(float[] values) {
        double norm = 0; for (float value : values) norm += value * value;
        norm = Math.sqrt(norm); if (norm == 0) return;
        for (int i = 0; i < values.length; i++) values[i] /= norm;
    }

    @Override public synchronized void close() {
        if (!running.getAndSet(false)) return;
        synchronized (frameLock) {
            if (pending != null) pending.bgr.release();
            pending = null;
            frameLock.notifyAll();
        }
        worker.interrupt();
        try { worker.join(10_000); }
        catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        if (worker.isAlive()) {
            // QNN calls are not interruptible. The worker owns cleanup and will
            // release its sessions as soon as that in-flight native call exits.
            android.util.Log.w("DeepLiveMobile", "Inference shutdown still waiting for native provider");
        }
    }

    private void releaseResources() {
        if (!resourcesReleased.compareAndSet(false, true)) return;
        softMask.release();
        runtime.close();
    }
}
