package dev.vcam.camlog;

import android.app.Application;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.util.Range;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.Supplier;

import de.robv.android.xposed.IXposedHookLoadPackage;
import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedBridge;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

/**
 * Global front-camera substitution module.
 *
 * <p>Every front-facing physical camera in every app is replaced with the
 * processed stream. Back, auxiliary, and external cameras stay enumerated and
 * open normally. On multi-front devices all front sensors are discovered
 * dynamically at enumeration time and redirected to the processed camera --
 * there is no per-app or per-device alias table. The processed camera takes
 * over the primary front's position in the public ID list; secondary fronts
 * remain visible but resolve to the processed camera when opened. The chosen
 * processed camera retains its real characteristics; only its LENS_FACING
 * value is presented as FRONT so selectors treat it as the front replacement.
 * Stream sizes and all other capabilities remain native to the processed
 * camera.</p>
 *
 * <p>The processed camera is never <em>added</em> to the enumerated list -- it
 * only ever replaces the primary front's slot, so the exposed list has one
 * fewer entry (the removed front), never one more.</p>
 *
 * <p>Virtual IDs: 100/101 (overlay mode) or 120 (single-camera mode). The
 * module detects which are present at runtime and falls back to unmodified
 * behavior if none exist, if no front camera is present, or if the operator's
 * kill-switch file {@code /data/local/tmp/vcam_disable} exists.</p>
 */
public final class CameraLoggerHooks implements IXposedHookLoadPackage {

    private static final String TAG = "VCamCamRoute";
    private static final String PROCESS_HOOK_MARKER =
            "dev.vcam.camroute.hooks_installed";
    private static final String PROCESS_EXCLUDED_MARKER =
            "dev.vcam.camroute.process_excluded";

    /** Processed camera IDs, preferred front first. */
    private static final String[] VIRTUAL_CANDIDATES = {"101", "100", "120"};

    /**
     * Presence of this file disables the module entirely, for every process,
     * without uninstalling or rebuilding. The native camera and microphone are
     * left completely untouched while it exists. Create it to fall back to the
     * real front camera globally; delete it to re-arm the processed front. It
     * lives under a world-readable tmp path so it can be toggled over adb
     * (`adb shell touch /data/local/tmp/vcam_disable`) or by the Arch manager.
     */
    private static final String KILL_SWITCH_PATH = "/data/local/tmp/vcam_disable";

    private static final int NATIVE_PERMISSION_CAPTURE_AUDIO_OUTPUT = 8;
    private static final int ANDROID_PER_USER_RANGE = 100_000;

    /** Last immutable route selected for this application process. */
    private volatile CameraRoutingPolicy.Route route =
            CameraRoutingPolicy.create(new String[0], null, VIRTUAL_CANDIDATES);

    /** Only these returned processed-camera metadata objects get facing spoofed. */
    private final Map<CameraCharacteristics, Boolean> processedCharacteristics =
            Collections.synchronizedMap(new WeakHashMap<>());

    /**
     * Cached characteristics of the real physical front camera that the
     * processed camera replaced.  Used to source FPS ranges and other
     * metadata that the external processed camera lacks but that apps
     * (especially full camera apps like Aperture) validate against.
     */
    private volatile CameraCharacteristics cachedFrontCharacteristics;

    /**
     * The camera device-map index of the real physical front camera.  Used
     * to redirect CamcorderProfile queries for the processed camera to the
     * real front's video quality profiles.  -1 when not yet discovered.
     */
    private volatile int cachedFrontDeviceIndex = -1;

    /** Prevent route discovery from recursively re-entering characteristics hooks. */
    private final ThreadLocal<Boolean> discovering =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** True only while this process has selected the processed front route. */
    private volatile boolean redirectedFrontActive;
    private final Map<MediaRecorder, Boolean> routedMediaRecorders =
            Collections.synchronizedMap(new WeakHashMap<>());
    private final Map<AudioRecord, Boolean> routedAudioRecords =
            Collections.synchronizedMap(new WeakHashMap<>());

    @Override
    public void handleLoadPackage(XC_LoadPackage.LoadPackageParam lpparam) {
        final String pkg = lpparam.packageName;
        final String process = lpparam.processName;
        if (killSwitchEngaged()) {
            // Global disable: leave the native camera and microphone untouched
            // in every process while the kill-switch file exists.
            markProcessExcluded();
            return;
        }
        if ("android".equals(pkg) && "android".equals(process)) {
            hookAudioServerCapturePermission(lpparam.classLoader);
            // The system_server process can emit additional load-package
            // callbacks for framework-hosted packages. Never install the
            // application camera/audio hooks into that shared process.
            markProcessExcluded();
            return;
        }
        if (shouldSkipPackage(pkg) || shouldSkipPackage(process)) {
            markProcessExcluded();
            return;
        }
        if (isProcessExcluded()) {
            return;
        }
        if (!claimHooksForProcess()) {
            XposedBridge.log(TAG + " [" + pkg
                    + "] hooks already installed in this process; skipping duplicate load");
            return;
        }
        final ClassLoader cl = lpparam.classLoader;
        // Every front-facing physical camera is discovered dynamically at
        // enumeration time and redirected to the processed stream, so the
        // module behaves as the global system front camera without any
        // per-package alias table.
        hookGetCameraIdList(cl, pkg);
        hookGetCameraCharacteristics(cl, pkg);
        hookCharacteristicsFacing(cl, pkg);
        hookOpenCamera(cl, pkg);
        hookMediaRecorder(cl, pkg);
        hookAudioRecord(cl, pkg);
        hookCamcorderProfile(cl, pkg);
    }

    /** True while the operator's global disable file is present. */
    private boolean killSwitchEngaged() {
        try {
            return new File(KILL_SWITCH_PATH).exists();
        } catch (Throwable t) {
            // A SecurityManager or unusual sandbox could deny the stat. Fail
            // closed toward the module's normal behavior (stay armed) rather
            // than silently disabling on an unrelated error.
            return false;
        }
    }

    /**
     * AudioFlinger protects Remote Submix with CAPTURE_AUDIO_OUTPUT even when
     * an ordinary recorder merely selects that device. The phone's native
     * permission controller receives a precomputed UID list from
     * AudioServerPermissionProvider, so changing a Java permission check in
     * the target app is too late. Extend that native-audio UID list for every
     * ordinary app UID (the module now acts as the global front camera), while
     * still excluding the framework, system UI, camera-extension, and this
     * module's own packages.
     *
     * <p>This grant does not reroute audio on its own. It only makes Remote
     * Submix selectable; the target-process hooks below actually select it,
     * and only while the processed front camera is open in that process. Rear
     * camera and non-camera sessions continue to use the native microphone.</p>
     */
    private void hookAudioServerCapturePermission(ClassLoader cl) {
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "com.android.server.audio.AudioServerPermissionProvider",
                cl,
                "getUidsHoldingPerm",
                int.class,
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        if (!Integer.valueOf(NATIVE_PERMISSION_CAPTURE_AUDIO_OUTPUT)
                                .equals(param.args[0])) {
                            return;
                        }
                        int[] original = (int[]) param.getResult();
                        Object packageMapObject = XposedHelpers.getObjectField(
                                param.thisObject, "mPackageMap");
                        Object userSupplierObject = XposedHelpers.getObjectField(
                                param.thisObject, "mUserIdSupplier");
                        if (!(packageMapObject instanceof Map)
                                || !(userSupplierObject instanceof Supplier)) {
                            XposedBridge.log(TAG
                                    + " [android] native audio permission state unavailable");
                            return;
                        }

                        @SuppressWarnings("unchecked")
                        Map<Object, Object> packageMap =
                                (Map<Object, Object>) packageMapObject;
                        Object userIdsObject = ((Supplier<?>) userSupplierObject).get();
                        if (!(userIdsObject instanceof int[])) {
                            return;
                        }

                        LinkedHashSet<Integer> uids = new LinkedHashSet<>();
                        if (original != null) {
                            for (int uid : original) uids.add(uid);
                        }
                        LinkedHashSet<String> matchedPackages = new LinkedHashSet<>();
                        for (Map.Entry<Object, Object> entry : packageMap.entrySet()) {
                            if (!(entry.getKey() instanceof Integer)
                                    || !(entry.getValue() instanceof Collection)) {
                                continue;
                            }
                            Collection<?> packages = (Collection<?>) entry.getValue();
                            boolean matched = false;
                            for (Object packageName : packages) {
                                // Grant to every ordinary app; skip only the
                                // framework/UI/module packages the camera hooks
                                // themselves never touch.
                                if (packageName instanceof String
                                        && !shouldSkipPackage((String) packageName)) {
                                    matchedPackages.add((String) packageName);
                                    matched = true;
                                }
                            }
                            if (!matched) continue;
                            int appId = (Integer) entry.getKey();
                            for (int userId : (int[]) userIdsObject) {
                                uids.add(userId * ANDROID_PER_USER_RANGE + appId);
                            }
                        }

                        int[] expanded = new int[uids.size()];
                        int index = 0;
                        for (Integer uid : uids) expanded[index++] = uid;
                        Arrays.sort(expanded);
                        param.setResult(expanded);
                        XposedBridge.log(TAG
                                + " [android] Remote Submix capture allowed for "
                                + matchedPackages.size()
                                + " front-redirectable app packages");
                    }
                }), "AudioServerPermissionProvider.getUidsHoldingPerm", "android");
    }

    private boolean shouldSkipPackage(String pkg) {
        return pkg == null
                || "android".equals(pkg)
                || "com.android.systemui".equals(pkg)
                || "com.android.cameraextensions".equals(pkg)
                || isPackageProcess(pkg, "dev.vcam.camlog")
                || isPackageProcess(pkg, "dev.vcam.app")
                || isPackageProcess(pkg, "dev.vcam.bridge")
                || isPackageProcess(pkg, "dev.vcam.mobile");
    }

    private boolean isPackageProcess(String value, String packageName) {
        return packageName.equals(value) || value.startsWith(packageName + ":");
    }

    private void markProcessExcluded() {
        synchronized (System.getProperties()) {
            System.setProperty(PROCESS_EXCLUDED_MARKER, Boolean.TRUE.toString());
        }
    }

    private boolean isProcessExcluded() {
        synchronized (System.getProperties()) {
            return Boolean.parseBoolean(System.getProperty(PROCESS_EXCLUDED_MARKER));
        }
    }

    /**
     * Vector can invoke the module repeatedly for an app, Google Play services,
     * and WebView inside the same PID. A Java system property is shared by all
     * module class loaders in that PID but is not shared across app processes,
     * making it a reliable process-local installation marker.
     */
    private boolean claimHooksForProcess() {
        synchronized (System.getProperties()) {
            if (Boolean.parseBoolean(System.getProperty(PROCESS_HOOK_MARKER))) {
                return false;
            }
            System.setProperty(PROCESS_HOOK_MARKER, Boolean.TRUE.toString());
            return true;
        }
    }

    private synchronized CameraRoutingPolicy.Route selectRoute(
            CameraManager manager, String[] ids) {
        if (Boolean.TRUE.equals(discovering.get())) {
            return route;
        }
        discovering.set(Boolean.TRUE);
        try {
            List<String> usableIds = new ArrayList<>(ids.length);
            List<String> rejectedIds = new ArrayList<>();
            // Every front-facing physical camera is a redirect target. The
            // first one found (enumeration order) is the primary front whose
            // list position the processed camera takes over; the rest stay
            // enumerated but resolve to the processed camera when opened. This
            // makes the module the global front camera on multi-front devices
            // without any per-app or per-device alias table.
            List<String> frontIds = new ArrayList<>();
            for (String id : ids) {
                try {
                    CameraCharacteristics characteristics =
                            manager.getCameraCharacteristics(id);
                    usableIds.add(id);
                    Integer facing = characteristics.get(
                            CameraCharacteristics.LENS_FACING);
                    if (!isProcessedCandidate(id)
                            && facing != null
                            && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                        frontIds.add(id);
                        // Cache the primary front's full characteristics so
                        // every query for the processed camera returns them
                        // instead of the external camera's limited metadata.
                        if (cachedFrontCharacteristics == null) {
                            cachedFrontCharacteristics = characteristics;
                            try {
                                cachedFrontDeviceIndex =
                                        Integer.parseInt(id);
                            } catch (NumberFormatException ignored) {
                                cachedFrontDeviceIndex = -1;
                            }
                        }
                    }
                } catch (Throwable t) {
                    rejectedIds.add(id);
                    XposedBridge.log(TAG + " camera " + id
                            + " was enumerated but its characteristics are unavailable; "
                            + "omitting it for this client: " + t);
                }
            }
            String defaultFrontId = frontIds.isEmpty() ? null : frontIds.get(0);
            String[] routableIds = usableIds.toArray(new String[0]);
            // The processed camera only ever REPLACES the primary front's slot;
            // it is never appended. createIdempotentWithAliases removes the
            // processed id's old position and reinserts it where the primary
            // front was, so the exposed list has exactly one fewer entry, not
            // one more. Secondary fronts remain visible but route on open.
            route = CameraRoutingPolicy.createIdempotentWithAliases(
                    routableIds, defaultFrontId,
                    frontIds.toArray(new String[0]), VIRTUAL_CANDIDATES);
            if (!rejectedIds.isEmpty()) {
                XposedBridge.log(TAG + " omitted unusable camera IDs " + rejectedIds);
            }
            return route;
        } finally {
            discovering.set(Boolean.FALSE);
        }
    }

    private boolean isProcessedCandidate(String id) {
        for (String candidate : VIRTUAL_CANDIDATES) {
            if (candidate.equals(id)) {
                return true;
            }
        }
        return false;
    }

    private void refreshRoute(CameraManager manager) {
        if (Boolean.TRUE.equals(discovering.get())) {
            return;
        }
        try {
            // The getCameraIdList after-hook performs discovery and publishes
            // the route. This also supports apps that open a remembered ID
            // without enumerating cameras during the current session.
            manager.getCameraIdList();
        } catch (Throwable t) {
            XposedBridge.log(TAG + " route discovery failed: " + t);
        }
    }

    private void hookGetCameraIdList(ClassLoader cl, final String pkg) {
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl, "getCameraIdList",
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        if (Boolean.TRUE.equals(discovering.get())) {
                            return;
                        }
                        Object result = param.getResult();
                        if (!(result instanceof String[])) {
                            return;
                        }
                        String[] ids = (String[]) result;
                        if (!(param.thisObject instanceof CameraManager)) {
                            return;
                        }
                        CameraRoutingPolicy.Route selected = selectRoute(
                                (CameraManager) param.thisObject, ids);
                        String[] exposed = selected.exposedIds();
                        // Always publish the validated list. This closes the race where
                        // a provider dies between enumeration and Persona's retry.
                        param.setResult(exposed);
                        if (selected.isActive()) {
                            XposedBridge.log(TAG + " [" + pkg
                                    + "] replaced default front "
                                    + selected.physicalFrontId()
                                    + " with processed " + selected.processedFrontId()
                                    + "; redirected fronts "
                                    + Arrays.toString(
                                            selected.redirectedPhysicalFrontIds())
                                    + ": " + Arrays.toString(ids) + " -> "
                                    + Arrays.toString(exposed));
                        } else if (selected.canRedirect()) {
                            XposedBridge.log(TAG + " [" + pkg
                                    + "] kept front aliases "
                                    + Arrays.toString(
                                            selected.redirectedPhysicalFrontIds())
                                    + " routed to processed "
                                    + selected.processedFrontId() + ": "
                                    + Arrays.toString(ids) + " -> "
                                    + Arrays.toString(exposed));
                        }
                    }
                }), "getCameraIdList", pkg);
    }

    private void hookGetCameraCharacteristics(ClassLoader cl, final String pkg) {
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl,
                "getCameraCharacteristics", String.class,
                new XC_MethodHook() {
                    @Override
                    protected void beforeHookedMethod(MethodHookParam param) {
                        if (!(param.thisObject instanceof CameraManager)
                                || !(param.args[0] instanceof String)
                                || Boolean.TRUE.equals(discovering.get())) {
                            return;
                        }
                        refreshRoute((CameraManager) param.thisObject);
                        CameraRoutingPolicy.Route selected = route;
                        String requested = (String) param.args[0];
                        String routed = selected.cameraIdForOpen(requested);
                        // Persona configures MediaRecorder immediately after
                        // resolving front-camera metadata, before openCamera.
                        // Arm the virtual mic here as well as at open time.
                        if (selected.isProcessedCameraId(routed)) {
                            setRedirectedFrontActive(true, pkg,
                                    "characteristics " + requested + " -> " + routed);
                        }
                        if (!requested.equals(routed)) {
                            param.args[0] = routed;
                            XposedBridge.log(TAG + " [" + pkg
                                    + "] characteristics " + requested + " -> " + routed);
                        }
                    }

                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        CameraRoutingPolicy.Route selected = route;
                        Object id = param.args[0];
                        Object result = param.getResult();
                        if (id instanceof String
                                && selected.isProcessedCameraId((String) id)
                                && result instanceof CameraCharacteristics) {
                            processedCharacteristics.put(
                                    (CameraCharacteristics) result,
                                    Boolean.TRUE);
                        }
                    }
                }), "getCameraCharacteristics", pkg);
    }

    private void hookCharacteristicsFacing(ClassLoader cl, final String pkg) {
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraCharacteristics", cl, "get",
                "android.hardware.camera2.CameraCharacteristics$Key",
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        if (!(param.thisObject instanceof CameraCharacteristics)
                                || !processedCharacteristics.containsKey(
                                        (CameraCharacteristics) param.thisObject)) {
                            return;
                        }
                        Object key = param.args[0];
                        // LENS_FACING → FRONT so camera selectors treat
                        // the processed camera as the front replacement.
                        if (CameraCharacteristics.LENS_FACING.equals(key)) {
                            param.setResult(
                                    CameraCharacteristics.LENS_FACING_FRONT);
                            return;
                        }
                        // AE_AVAILABLE_TARGET_FPS_RANGES — the external camera
                        // only reports [15,30] (variable), missing the [30,30]
                        // fixed range that CameraX/Aperture requires for video.
                        // Inject the real front camera's full FPS range set.
                        if (CameraCharacteristics
                                .CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                                .equals(key)) {
                            CameraCharacteristics front =
                                    cachedFrontCharacteristics;
                            if (front != null) {
                                @SuppressWarnings("unchecked")
                                Range<Integer>[] frontRanges =
                                        (Range<Integer>[]) front.get(
                                                CameraCharacteristics
                                                .CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
                                if (frontRanges != null
                                        && frontRanges.length > 0) {
                                    param.setResult(frontRanges);
                                }
                            }
                            return;
                        }
                        // INFO_SUPPORTED_HARDWARE_LEVEL — EXTERNAL (4) causes
                        // CameraX to take a restricted codepath that rejects
                        // video quality profiles. Report LIMITED (0) so apps
                        // treat the processed camera like a normal built-in.
                        if (CameraCharacteristics
                                .INFO_SUPPORTED_HARDWARE_LEVEL.equals(key)) {
                            param.setResult(
                                    CameraCharacteristics
                                    .INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED);
                            return;
                        }
                    }
                }), "CameraCharacteristics.get(spoofed keys)", pkg);
    }

    private void hookOpenCamera(ClassLoader cl, final String pkg) {
        final XC_MethodHook redirect = new XC_MethodHook() {
            @Override
            protected void beforeHookedMethod(MethodHookParam param) {
                Object id = param.args[0];
                if (!(id instanceof String)) {
                    return;
                }
                if (param.thisObject instanceof CameraManager) {
                    refreshRoute((CameraManager) param.thisObject);
                }
                String routed = route.cameraIdForOpen((String) id);
                final boolean processedFront = route.isProcessedCameraId(routed);
                setRedirectedFrontActive(processedFront, pkg,
                        "openCamera " + id + " -> " + routed);
                if (!id.equals(routed)) {
                    XposedBridge.log(TAG + " [" + pkg + "] openCamera "
                            + id + " -> " + routed);
                    param.args[0] = routed;
                }
                if (processedFront) {
                    wrapStateCallback(param, pkg);
                }
            }
        };
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl, "openCamera",
                String.class,
                "android.hardware.camera2.CameraDevice$StateCallback",
                android.os.Handler.class,
                redirect), "openCamera(Handler)", pkg);
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl, "openCamera",
                String.class,
                java.util.concurrent.Executor.class,
                "android.hardware.camera2.CameraDevice$StateCallback",
                redirect), "openCamera(Executor)", pkg);
    }

    private void wrapStateCallback(XC_MethodHook.MethodHookParam param,
            final String pkg) {
        for (int index = 1; index < param.args.length; index++) {
            Object argument = param.args[index];
            if (!(argument instanceof CameraDevice.StateCallback)) {
                continue;
            }
            final CameraDevice.StateCallback delegate =
                    (CameraDevice.StateCallback) argument;
            param.args[index] = new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    delegate.onOpened(camera);
                }

                @Override
                public void onClosed(CameraDevice camera) {
                    setRedirectedFrontActive(false, pkg,
                            "processed camera " + camera.getId() + " closed");
                    delegate.onClosed(camera);
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    setRedirectedFrontActive(false, pkg,
                            "processed camera " + camera.getId() + " disconnected");
                    delegate.onDisconnected(camera);
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    setRedirectedFrontActive(false, pkg,
                            "processed camera " + camera.getId() + " error=" + error);
                    delegate.onError(camera, error);
                }
            };
            return;
        }
    }

    private void hookMediaRecorder(ClassLoader cl, final String pkg) {
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.MediaRecorder", cl, "prepare",
                new XC_MethodHook() {
                    @Override
                    protected void beforeHookedMethod(MethodHookParam param) {
                        if (!redirectedFrontActive
                                || !(param.thisObject instanceof MediaRecorder)) {
                            return;
                        }
                        MediaRecorder recorder = (MediaRecorder) param.thisObject;
                        AudioDeviceInfo input = findRemoteSubmixInput(pkg);
                        if (input != null && recorder.setPreferredDevice(input)) {
                            routedMediaRecorders.put(recorder, Boolean.TRUE);
                            XposedBridge.log(TAG + " [" + pkg
                                    + "] MediaRecorder -> Remote Submix In"
                                    + " for redirected front camera");
                        } else {
                            XposedBridge.log(TAG + " [" + pkg
                                    + "] MediaRecorder virtual-mic route unavailable;"
                                    + " preserving native microphone");
                        }
                    }
                }), "MediaRecorder.prepare", pkg);
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.MediaRecorder", cl, "release",
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        if (param.thisObject instanceof MediaRecorder) {
                            routedMediaRecorders.remove((MediaRecorder) param.thisObject);
                        }
                    }
                }), "MediaRecorder.release", pkg);
    }

    private void hookAudioRecord(ClassLoader cl, final String pkg) {
        XC_MethodHook routeBeforeStart = new XC_MethodHook() {
            @Override
            protected void beforeHookedMethod(MethodHookParam param) {
                if (!redirectedFrontActive
                        || !(param.thisObject instanceof AudioRecord)) {
                    return;
                }
                AudioRecord record = (AudioRecord) param.thisObject;
                AudioDeviceInfo input = findRemoteSubmixInput(pkg);
                if (input != null && record.setPreferredDevice(input)) {
                    routedAudioRecords.put(record, Boolean.TRUE);
                    XposedBridge.log(TAG + " [" + pkg
                            + "] AudioRecord -> Remote Submix In"
                            + " for redirected front camera");
                } else {
                    XposedBridge.log(TAG + " [" + pkg
                            + "] AudioRecord virtual-mic route unavailable;"
                            + " preserving native microphone");
                }
            }
        };
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.AudioRecord", cl, "startRecording",
                routeBeforeStart), "AudioRecord.startRecording()", pkg);
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.AudioRecord", cl, "startRecording",
                "android.media.MediaSyncEvent", routeBeforeStart),
                "AudioRecord.startRecording(MediaSyncEvent)", pkg);
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.AudioRecord", cl, "release",
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param) {
                        if (param.thisObject instanceof AudioRecord) {
                            routedAudioRecords.remove((AudioRecord) param.thisObject);
                        }
                    }
                }), "AudioRecord.release", pkg);
    }

    /**
     * Redirect CamcorderProfile queries for the processed camera to the real
     * front camera's device-map index.  Without this, {@code hasProfile(120,
     * quality)} returns false for every quality (external cameras have no
     * CamcorderProfile entries), and CameraX rejects all video qualities with
     * "Video frame rate not supported".
     *
     * <p>Both {@code hasProfile(int, int)} and {@code get(int, int)} are
     * redirected so the app reads the real front's profiles (which include
     * 720p and below) and configures a compatible session.</p>
     */
    private void hookCamcorderProfile(ClassLoader cl, final String pkg) {
        final XC_MethodHook redirectId = new XC_MethodHook() {
            @Override
            protected void beforeHookedMethod(MethodHookParam param) {
                CameraRoutingPolicy.Route selected = route;
                if (!selected.canRedirect() || cachedFrontDeviceIndex < 0) {
                    return;
                }
                Object rawId = param.args[0];
                if (!(rawId instanceof Integer)) {
                    return;
                }
                int requestedId = (Integer) rawId;
                // Map the processed camera's device-map index to the real
                // front's.  The processed camera is external, so its numeric
                // id is typically 120 (100+ offset); the real front is 1.
                String processedFrontId = selected.processedFrontId();
                if (processedFrontId != null) {
                    try {
                        if (requestedId == Integer.parseInt(processedFrontId)) {
                            param.args[0] = cachedFrontDeviceIndex;
                        }
                    } catch (NumberFormatException ignored) {
                        // Non-numeric processed camera ID; skip redirect.
                    }
                }
            }
        };
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.CamcorderProfile", cl, "hasProfile",
                int.class, int.class, redirectId),
                "CamcorderProfile.hasProfile", pkg);
        safeHook(() -> XposedHelpers.findAndHookMethod(
                "android.media.CamcorderProfile", cl, "get",
                int.class, int.class, redirectId),
                "CamcorderProfile.get", pkg);
    }

    private AudioDeviceInfo findRemoteSubmixInput(String pkg) {
        try {
            // Do not reference AndroidAppHelper here. Some modern Xposed
            // implementations relocate that API class for legacy modules,
            // which can leave a rewritten-but-missing runtime reference.
            // ActivityThread belongs to the Android process and remains
            // stable regardless of which Xposed implementation loads us.
            Class<?> activityThread = XposedHelpers.findClass(
                    "android.app.ActivityThread", null);
            Application application = (Application) XposedHelpers.callStaticMethod(
                    activityThread, "currentApplication");
            if (application == null) {
                XposedBridge.log(TAG + " [" + pkg
                        + "] application unavailable while resolving virtual mic");
                return null;
            }
            AudioManager manager = application.getSystemService(AudioManager.class);
            if (manager == null) {
                return null;
            }
            for (AudioDeviceInfo device :
                    manager.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
                if (device.getType() == AudioDeviceInfo.TYPE_REMOTE_SUBMIX
                        && device.isSource()
                        && "0".equals(device.getAddress())) {
                    return device;
                }
            }
        } catch (Throwable t) {
            XposedBridge.log(TAG + " [" + pkg
                    + "] virtual-mic discovery failed: " + t);
        }
        return null;
    }

    private synchronized void setRedirectedFrontActive(boolean active,
            String pkg, String reason) {
        if (redirectedFrontActive == active) {
            return;
        }
        redirectedFrontActive = active;
        XposedBridge.log(TAG + " [" + pkg + "] virtual microphone "
                + (active ? "armed" : "disarmed") + ": " + reason);
        if (!active) {
            clearPreferredInputRoutes(pkg);
        }
    }

    private void clearPreferredInputRoutes(String pkg) {
        synchronized (routedMediaRecorders) {
            for (MediaRecorder recorder :
                    new ArrayList<>(routedMediaRecorders.keySet())) {
                try {
                    recorder.setPreferredDevice(null);
                } catch (Throwable ignored) {
                    // The recorder may already have been released by the app.
                }
            }
            routedMediaRecorders.clear();
        }
        synchronized (routedAudioRecords) {
            for (AudioRecord record : new ArrayList<>(routedAudioRecords.keySet())) {
                try {
                    record.setPreferredDevice(null);
                } catch (Throwable ignored) {
                    // The record may already have been released by the app.
                }
            }
            routedAudioRecords.clear();
        }
        XposedBridge.log(TAG + " [" + pkg
                + "] native microphone routing restored");
    }

    private interface HookInstaller {
        void install() throws Throwable;
    }

    private void safeHook(HookInstaller installer, String what, String pkg) {
        try {
            installer.install();
        } catch (Throwable t) {
            XposedBridge.log(TAG + " [" + pkg + "] hook " + what + " skipped: " + t);
        }
    }
}
