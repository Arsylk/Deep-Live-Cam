package dev.vcam.mobile;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Debug;
import android.os.Environment;
import android.os.PowerManager;
import android.provider.Settings;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import java.io.File;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.opencv.core.Mat;

public final class MainActivity extends Activity implements
        CameraController.Listener, FacePipeline.Listener {
    private static final int CAMERA_PERMISSION = 100;
    private static final int PICK_SOURCE = 101;
    private final ExecutorService background = Executors.newSingleThreadExecutor();
    private final Object lifecycleLock = new Object();
    private final AtomicBoolean processingEnabled = new AtomicBoolean(true);
    private PreviewView preview;
    private TextView stateView, engineView, sourceView, metricsView, recordView;
    private LinearLayout historyLayout;
    private Switch processingSwitch;
    private Button lensButton, recordButton;
    private ModelStore modelStore;
    private SourceRepository sources;
    private volatile FacePipeline pipeline;
    private volatile boolean destroyed;
    private volatile boolean cameraLive;
    private boolean visible;
    private boolean modelInitializationInProgress;
    private boolean controlsEnabled = true;
    private CameraController camera;
    private final VideoRecorder recorder = new VideoRecorder();
    private volatile boolean comparing;
    private volatile double cameraFps;
    private volatile long cameraDropped;
    private volatile int frameWidth = 720, frameHeight = 1280;
    private long lastMetricsUi;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        System.loadLibrary("opencv_java4");
        modelStore = new ModelStore(this);
        sources = new SourceRepository(this);
        buildUi();
        refreshHistory();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        }
    }

    private void buildUi() {
        int pad = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(8, 17, 16));
        root.setPadding(pad, pad, pad, pad);
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
                view.setPadding(pad + bars.left, pad + bars.top, pad + bars.right, pad + bars.bottom);
            } else {
                view.setPadding(pad + insets.getSystemWindowInsetLeft(),
                        pad + insets.getSystemWindowInsetTop(),
                        pad + insets.getSystemWindowInsetRight(),
                        pad + insets.getSystemWindowInsetBottom());
            }
            return insets;
        });

        TextView title = text("Deep Live Mobile", 24, Color.WHITE);
        root.addView(title, matchWrap());
        stateView = text("Starting offline prototype…", 14, Color.rgb(158, 220, 211));
        root.addView(stateView, matchWrap());
        engineView = text("Inference: checking local model pack…", 12, Color.rgb(182, 201, 198));
        engineView.setContentDescription("On-device inference status");
        root.addView(engineView, matchWrap());

        preview = new PreviewView(this);
        preview.setContentDescription("Processed camera preview");
        GradientDrawable frame = new GradientDrawable();
        frame.setColor(Color.BLACK); frame.setCornerRadius(dp(16)); frame.setStroke(dp(1), Color.rgb(48, 90, 84));
        preview.setBackground(frame); preview.setClipToOutline(true);
        root.addView(preview, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        LinearLayout controls = row();
        processingSwitch = new Switch(this); processingSwitch.setText("Face swap"); processingSwitch.setTextColor(Color.WHITE);
        processingSwitch.setChecked(true); processingSwitch.setContentDescription("Toggle face swap processing");
        processingSwitch.setOnCheckedChangeListener((button, checked) -> processingEnabled.set(checked));
        controls.addView(processingSwitch, weighted());
        Button compare = button("Hold raw"); compare.setContentDescription("Hold to compare raw camera");
        compare.setOnTouchListener((view, event) -> {
            comparing = event.getAction() != MotionEvent.ACTION_UP && event.getAction() != MotionEvent.ACTION_CANCEL;
            return true;
        }); controls.addView(compare);
        lensButton = button("Front"); lensButton.setContentDescription("Switch physical camera");
        lensButton.setOnClickListener(v -> { if (camera != null) { camera.switchLens(); lensButton.setText(camera.isFront() ? "Front" : "Rear"); } });
        controls.addView(lensButton); root.addView(controls, matchWrap());

        LinearLayout sourceHeader = row();
        Button add = button("Add source"); add.setContentDescription("Choose local source picture"); add.setOnClickListener(v -> pickSource());
        sourceHeader.addView(add);
        add.setTag("requires-camera");
        sourceView = text("No source selected", 13, Color.LTGRAY); sourceHeader.addView(sourceView, weighted());
        root.addView(sourceHeader, matchWrap());
        HorizontalScrollView historyScroll = new HorizontalScrollView(this);
        historyScroll.setHorizontalScrollBarEnabled(false);
        historyLayout = row(); historyLayout.setContentDescription("Recent source pictures");
        historyScroll.addView(historyLayout); root.addView(historyScroll, new LinearLayout.LayoutParams(-1, dp(78)));

        LinearLayout recording = row();
        recordButton = button("Record"); recordButton.setContentDescription("Start or stop local preview recording");
        recordButton.setTag("requires-camera");
        recordButton.setOnClickListener(v -> toggleRecording()); recording.addView(recordButton);
        recordView = text("Not recording", 12, Color.LTGRAY); recording.addView(recordView, weighted());
        root.addView(recording, matchWrap());

        metricsView = text("Metrics: waiting for frames", 12, Color.rgb(182, 201, 198));
        metricsView.setContentDescription("Live processing metrics");
        root.addView(metricsView, matchWrap());
        setContentView(root);
    }

    private void initializeModels() {
        if (!modelStore.ready()) { engineView.setText("Inference unavailable: " + modelStore.readiness()); return; }
        synchronized (lifecycleLock) {
            if (destroyed || !visible || !cameraLive
                    || pipeline != null || modelInitializationInProgress) return;
            modelInitializationInProgress = true;
        }
        engineView.setText("Inference: verifying offline model pack…");
        background.execute(() -> {
            FacePipeline ready = null;
            try {
                modelStore.verify();
                ready = new FacePipeline(modelStore, this);
                SourceRepository.Entry selected = sources.selected();
                if (selected != null) sources.select(selected, ready);
                boolean shouldPublish;
                synchronized (lifecycleLock) {
                    modelInitializationInProgress = false;
                    shouldPublish = !destroyed && visible && cameraLive;
                    if (shouldPublish) pipeline = ready;
                }
                if (!shouldPublish) { ready.close(); return; }
                FacePipeline published = ready;
                ready = null;
                runOnUiThread(() -> {
                    if (destroyed || !visible || pipeline != published) return;
                    engineView.setText("Inference ready: " + published.provider());
                    refreshHistory();
                });
            } catch (Throwable error) {
                if (ready != null) ready.close();
                synchronized (lifecycleLock) { modelInitializationInProgress = false; }
                runOnUiThread(() -> {
                    if (!destroyed && visible) engineView.setText("Inference failed: " + error.getMessage());
                });
            }
        });
    }

    private void pickSource() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE); intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        startActivityForResult(intent, PICK_SOURCE);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_SOURCE || resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData(); FacePipeline active = pipeline;
        if (uri == null || active == null) { sourceView.setText("Wait for inference models before adding a source"); return; }
        sourceView.setText("Analyzing source locally…");
        background.execute(() -> {
            try {
                SourceRepository.Entry entry = sources.importLocal(uri, active);
                runOnUiThread(() -> { refreshHistory(); sourceView.setText(entry.label); });
            } catch (Throwable error) {
                runOnUiThread(() -> sourceView.setText("Source rejected: " + error.getMessage()));
            }
        });
    }

    private void refreshHistory() {
        if (historyLayout == null) return;
        historyLayout.removeAllViews(); SourceRepository.Entry selected = sources.selected();
        for (SourceRepository.Entry entry : sources.entries()) {
            ImageButton item = new ImageButton(this);
            item.setImageBitmap(BitmapFactory.decodeFile(entry.thumbnail.getAbsolutePath()));
            item.setScaleType(ImageButton.ScaleType.CENTER_CROP); item.setContentDescription("Use source " + entry.label);
            GradientDrawable backgroundDrawable = new GradientDrawable(); backgroundDrawable.setColor(Color.rgb(22, 35, 33));
            backgroundDrawable.setCornerRadius(dp(10));
            backgroundDrawable.setStroke(dp(selected != null && selected.id.equals(entry.id) ? 3 : 1),
                    selected != null && selected.id.equals(entry.id) ? Color.rgb(100, 216, 203) : Color.rgb(60, 85, 81));
            item.setBackground(backgroundDrawable);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(68), dp(68)); params.setMargins(0, dp(4), dp(8), dp(4));
            historyLayout.addView(item, params);
            item.setOnClickListener(v -> selectSource(entry));
        }
        sourceView.setText(selected == null ? "No source selected" : selected.label);
    }

    private void selectSource(SourceRepository.Entry entry) {
        FacePipeline active = pipeline; if (active == null) return;
        sourceView.setText("Loading " + entry.label + "…");
        background.execute(() -> {
            try { sources.select(entry, active); runOnUiThread(this::refreshHistory); }
            catch (Throwable error) { runOnUiThread(() -> sourceView.setText("Source failed: " + error.getMessage())); }
        });
    }

    private void toggleRecording() {
        if (recorder.active()) {
            File output = recorder.output(); recorder.close(); recordButton.setText("Record");
            recordView.setText("Saved " + (output == null ? "recording" : output.getAbsolutePath())); return;
        }
        try {
            File base = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
            recorder.start(new File(base, "DeepLiveMobile"), frameWidth, frameHeight);
            recordButton.setText("Stop"); recordView.setText("Recording exact displayed mode…");
        } catch (Exception error) { recordView.setText("Recording unavailable: " + error.getMessage()); }
    }

    @Override protected void onStart() {
        super.onStart();
        synchronized (lifecycleLock) { visible = true; }
        if (pipeline == null) engineView.setText("Inference standby: waiting for a free physical camera");
    }

    @Override protected void onResume() {
        super.onResume();
        startCameraIfPermitted();
    }

    private void startCameraIfPermitted() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED && camera == null) {
            camera = new CameraController(this, this); camera.start();
        }
    }

    @Override protected void onPause() {
        cameraLive = false;
        if (camera != null) { camera.close(); camera = null; }
        if (recorder.active()) { recorder.close(); recordButton.setText("Record"); recordView.setText("Recording stopped on background"); }
        super.onPause();
    }

    @Override protected void onStop() {
        FacePipeline inactive;
        synchronized (lifecycleLock) {
            visible = false;
            inactive = pipeline;
            pipeline = null;
        }
        if (inactive != null) {
            engineView.setText("Inference suspended while app is in background");
            background.execute(inactive::close);
        }
        super.onStop();
    }

    @Override protected void onDestroy() {
        FacePipeline active;
        synchronized (lifecycleLock) {
            destroyed = true;
            active = pipeline;
            pipeline = null;
        }
        if (active != null) background.execute(active::close);
        background.shutdown();
        preview.release(); recorder.close(); super.onDestroy();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) startCameraIfPermitted();
        else if (requestCode == CAMERA_PERMISSION) stateView.setText("Camera permission denied; enable it in system settings");
    }

    @Override public void onCameraFrame(Mat bgr, long timestampNs, long generation) {
        frameWidth = bgr.cols(); frameHeight = bgr.rows(); FacePipeline active = pipeline;
        boolean process = processingEnabled.get() && active != null && active.sourceReady();
        if (process) active.offer(bgr, timestampNs, generation);
        if (!process || comparing) {
            preview.show(bgr); recordFrameIfActive(bgr, timestampNs);
        }
    }

    @Override public void onProcessed(Mat bgr, long timestampNs, long cameraGeneration, long sourceGeneration) {
        try {
            CameraController current = camera; FacePipeline active = pipeline;
            if (current == null || active == null || current.generation() != cameraGeneration
                    || active.sourceGeneration() != sourceGeneration || !processingEnabled.get()) return;
            if (!comparing) preview.show(bgr);
            if (!comparing) recordFrameIfActive(bgr, timestampNs);
        } finally { bgr.release(); }
    }

    private void recordFrameIfActive(Mat frame, long timestampNs) {
        if (!recorder.active() || recorder.frame(frame, timestampNs)) return;
        runOnUiThread(() -> {
            if (destroyed) return;
            recordButton.setText("Record");
            recordView.setText("Recording stopped after an encoder error");
        });
    }

    @Override public void onCameraState(String state) {
        cameraLive = state.startsWith("Live ");
        runOnUiThread(() -> {
            if (destroyed) return;
            stateView.setText(state);
            setCameraControlsEnabled(cameraLive);
            if (cameraLive) initializeModels();
        });
    }

    private void setCameraControlsEnabled(boolean enabled) {
        if (controlsEnabled == enabled) return;
        controlsEnabled = enabled;
        processingSwitch.setEnabled(enabled);
        lensButton.setEnabled(enabled);
        for (int i = 0; i < ((ViewGroup) processingSwitch.getParent()).getChildCount(); i++) {
            View child = ((ViewGroup) processingSwitch.getParent()).getChildAt(i);
            if (child instanceof Button) child.setEnabled(enabled);
        }
        ViewGroup root = (ViewGroup) recordButton.getRootView();
        setTaggedControlsEnabled(root, enabled);
    }

    private void setTaggedControlsEnabled(View view, boolean enabled) {
        if ("requires-camera".equals(view.getTag())) view.setEnabled(enabled);
        if (!(view instanceof ViewGroup)) return;
        ViewGroup group = (ViewGroup) view;
        for (int i = 0; i < group.getChildCount(); i++) {
            setTaggedControlsEnabled(group.getChildAt(i), enabled);
        }
    }
    @Override public void onCameraMetrics(double fps, long dropped) {
        cameraFps = fps; cameraDropped = dropped;
        FacePipeline active = pipeline;
        if (active == null || !active.sourceReady() || !processingEnabled.get()) showCameraOnlyMetrics();
    }
    @Override public void onError(String detail) {
        processingEnabled.set(false);
        runOnUiThread(() -> {
            if (destroyed) return;
            processingSwitch.setChecked(false);
            engineView.setText("Inference paused; raw preview active: " + detail);
        });
    }

    @Override public void onMetrics(FacePipeline.Metrics value) {
        long now = android.os.SystemClock.elapsedRealtime(); if (now - lastMetricsUi < 500) return; lastMetricsUi = now;
        PowerManager power = getSystemService(PowerManager.class); int thermal = power.getCurrentThermalStatus();
        long pss = Debug.getPss();
        String text = String.format(Locale.US,
                "Camera %.1f FPS · inference %.2f FPS · face %s\n" +
                "detect %.1f ms · swap %.1f ms · total %.1f ms · dropped %d/%d\n" +
                "%s · PSS %.0f MiB · thermal %d · tracker adaptive EMA/fade",
                cameraFps, value.inferenceFps, value.faceFound ? "tracked" : "not found",
                value.detectionMs, value.swapMs, value.totalMs, value.dropped, cameraDropped,
                value.provider, pss / 1024.0, thermal);
        runOnUiThread(() -> metricsView.setText(text));
    }

    private void showCameraOnlyMetrics() {
        PowerManager power = getSystemService(PowerManager.class);
        long pss = Debug.getPss();
        String mode = pipeline == null ? "models loading" : "raw preview";
        String text = String.format(Locale.US,
                "Camera %.1f FPS · %s · dropped %d\nPSS %.0f MiB · thermal %d · fully offline",
                cameraFps, mode, cameraDropped, pss / 1024.0, power.getCurrentThermalStatus());
        runOnUiThread(() -> metricsView.setText(text));
    }

    private Button button(String label) { Button result = new Button(this); result.setText(label); result.setAllCaps(false); return result; }
    private TextView text(String value, float size, int color) { TextView result = new TextView(this); result.setText(value); result.setTextSize(size); result.setTextColor(color); result.setPadding(0, dp(4), dp(8), dp(4)); return result; }
    private LinearLayout row() { LinearLayout result = new LinearLayout(this); result.setOrientation(LinearLayout.HORIZONTAL); result.setGravity(Gravity.CENTER_VERTICAL); return result; }
    private LinearLayout.LayoutParams weighted() { return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f); }
    private LinearLayout.LayoutParams matchWrap() { return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
}
