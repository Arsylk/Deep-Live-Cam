package dev.vcam.mobile;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

final class InferenceRuntime implements AutoCloseable {
    private static final String TAG = "DeepLiveMobile";
    enum Provider { QNN_GPU, CPU }

    private final OrtEnvironment environment;
    private final ModelStore models;
    private OrtSession detector;
    private OrtSession swapper;
    private boolean qnnBackendAvailable;
    private boolean detectorOnQnn;
    private boolean swapperOnQnn;
    private boolean swapperFp16;

    InferenceRuntime(ModelStore models) throws Exception {
        this.models = models;
        environment = OrtEnvironment.getEnvironment("deep-live-mobile");
        environment.setTelemetry(false);
        android.util.Log.i(TAG, "ORT " + environment.getVersion() + " providers="
                + OrtEnvironment.getAvailableProviders());
        openPersistentSessions();
    }

    String provider() {
        String swap = swapperFp16 ? "QNN GPU FP16" : "QNN GPU FP32";
        if (detectorOnQnn && swapperOnQnn) {
            return swapperFp16 ? "QNN GPU: detector FP32 + swap FP16" : "QNN GPU FP32: detector + swap";
        }
        if (detectorOnQnn) return "QNN detector + CPU swap";
        if (swapperOnQnn) return "CPU detector + " + swap + " swap";
        return "CPU";
    }

    private void openPersistentSessions() throws Exception {
        try {
            // ORT can return a valid session even when QNN initialization failed
            // and every node fell back to CPU. A tiny, fully-QNN-compatible graph
            // with CPU fallback disabled makes backend detection unambiguous.
            try (OrtSession ignored = open(models.file(ModelStore.QNN_PROBE), Provider.QNN_GPU, true)) {}
            qnnBackendAvailable = true;
        } catch (Exception failure) {
            android.util.Log.w(TAG, "QNN GPU probe failed; using CPU", failure);
        }
        if (qnnBackendAvailable) {
            try {
                detector = open(models.file(ModelStore.DETECTOR), Provider.QNN_GPU, true);
                detectorOnQnn = true;
            } catch (Exception failure) {
                android.util.Log.w(TAG, "Detector is not fully QNN-compatible; using CPU", failure);
            }
            try {
                swapper = open(models.file(ModelStore.SWAPPER_FP16), Provider.QNN_GPU, true);
                swapperOnQnn = true;
                swapperFp16 = true;
            } catch (Exception failure) {
                android.util.Log.w(TAG, "FP16 swapper is not fully QNN-compatible", failure);
                try {
                    swapper = open(models.file(ModelStore.SWAPPER), Provider.QNN_GPU, true);
                    swapperOnQnn = true;
                } catch (Exception fp32Failure) {
                    android.util.Log.w(TAG, "FP32 swapper is not fully QNN-compatible; using CPU", fp32Failure);
                }
            }
        }
        if (detector == null) detector = open(models.file(ModelStore.DETECTOR), Provider.CPU, false);
        if (swapper == null) swapper = open(models.file(ModelStore.SWAPPER), Provider.CPU, false);
        android.util.Log.i(TAG, "Persistent inference provider=" + provider());
    }

    private OrtSession open(File path, Provider requested, boolean strict) throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setInterOpNumThreads(1);
        options.setIntraOpNumThreads(requested == Provider.CPU ? 4 : 1);
        // Strictly delegated fixed-shape QNN graphs do not use ORT's CPU
        // memory pattern or arena for graph execution. Disabling both avoids
        // retaining redundant host-side high-water allocations.
        if (requested == Provider.QNN_GPU) {
            options.setMemoryPatternOptimization(false);
            options.setCPUArenaAllocator(false);
        }
        if (requested == Provider.QNN_GPU) {
            options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
            Map<String, String> config = new HashMap<>();
            config.put("backend_type", "gpu");
            config.put("profiling_level", "off");
            options.addQnn(config);
            if (strict) options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        try {
            return environment.createSession(path.getAbsolutePath(), options);
        } finally {
            options.close();
        }
    }

    synchronized float[][] detect(float[] input) throws Exception {
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(input), new long[]{1, 3, 640, 640});
             OrtSession.Result result = detector.run(Collections.singletonMap("input.1", tensor))) {
            float[][] values = new float[result.size()][];
            for (int i = 0; i < result.size(); i++) values[i] = flatten(result.get(i).getValue());
            return values;
        }
    }

    synchronized float[] recognize(float[] input) throws Exception {
        OrtSession recognition = null;
        if (qnnBackendAvailable) {
            try {
                recognition = open(models.file(ModelStore.RECOGNIZER), Provider.QNN_GPU, true);
            } catch (Exception failure) {
                android.util.Log.w(TAG, "Recognizer is not fully QNN-compatible; using CPU", failure);
            }
        }
        if (recognition == null) recognition = open(models.file(ModelStore.RECOGNIZER), Provider.CPU, false);
        try (OrtSession session = recognition;
             OnnxTensor tensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(input), new long[]{1, 3, 112, 112});
             OrtSession.Result result = session.run(Collections.singletonMap("input.1", tensor))) {
            return flatten(result.get(0).getValue());
        }
    }

    synchronized float[] swap(float[] target, float[] latent) throws Exception {
        try (OnnxTensor targetTensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(target), new long[]{1, 3, 128, 128});
             OnnxTensor sourceTensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(latent), new long[]{1, 512})) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("target", targetTensor);
            inputs.put("source", sourceTensor);
            try (OrtSession.Result result = swapper.run(inputs)) {
                return flatten(result.get(0).getValue());
            }
        }
    }

    private static float[] flatten(Object value) {
        if (value instanceof float[]) return (float[]) value;
        int count = count(value);
        float[] output = new float[count];
        fill(value, output, new int[]{0});
        return output;
    }

    private static int count(Object value) {
        if (value instanceof float[]) return ((float[]) value).length;
        if (value instanceof Object[]) {
            int count = 0;
            for (Object item : (Object[]) value) count += count(item);
            return count;
        }
        throw new IllegalArgumentException("unexpected tensor value " + value.getClass());
    }

    private static void fill(Object value, float[] output, int[] offset) {
        if (value instanceof float[]) {
            float[] items = (float[]) value;
            System.arraycopy(items, 0, output, offset[0], items.length);
            offset[0] += items.length;
        } else {
            for (Object item : (Object[]) value) fill(item, output, offset);
        }
    }

    @Override public synchronized void close() {
        closeQuietly(detector); closeQuietly(swapper);
        detector = null; swapper = null;
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) return;
        try { closeable.close(); } catch (Exception ignored) {}
    }
}
