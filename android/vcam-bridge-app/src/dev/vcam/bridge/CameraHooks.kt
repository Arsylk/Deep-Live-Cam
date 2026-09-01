package dev.vcam.bridge

import java.util.Arrays
import java.util.HashSet

import de.robv.android.xposed.IXposedHookLoadPackage
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XposedBridge
import de.robv.android.xposed.XposedHelpers
import de.robv.android.xposed.callbacks.XC_LoadPackage

/**
 * Xposed hooks: route all camera access to the processed cameras.
 *
 * - getCameraIdList() returns only virtual camera IDs (100, 101, or 120)
 * - openCamera(id) on a non-virtual ID is redirected to the preferred front ID
 *
 * Falls back to unmodified behavior if no virtual cameras are present.
 */
class CameraHooks : IXposedHookLoadPackage {

    companion object {
        private const val TAG = "VCamHooks"
        private val VIRTUAL_CANDIDATES = arrayOf("101", "100", "120")
    }

    private val virtualIds = HashSet<String>()
    private var preferredFrontId: String? = null

    override fun handleLoadPackage(lpparam: XC_LoadPackage.LoadPackageParam) {
        val pkg = lpparam.packageName
        val cl = lpparam.classLoader
        hookGetCameraIdList(cl, pkg)
        hookOpenCamera(cl, pkg)
    }

    private fun selectVirtual(ids: Array<String>) {
        virtualIds.clear()
        preferredFrontId = null
        val present = HashSet(Arrays.asList(*ids))
        for (cand in VIRTUAL_CANDIDATES) {
            if (present.contains(cand)) {
                virtualIds.add(cand)
                if (preferredFrontId == null) preferredFrontId = cand
            }
        }
    }

    private fun hookGetCameraIdList(cl: ClassLoader, pkg: String) {
        safeHook("getCameraIdList", pkg) {
            XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl, "getCameraIdList",
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val result = param.getResult()
                        if (result !is Array<*>) return
                        val ids = result as Array<String>
                        if (virtualIds.isEmpty()) selectVirtual(ids)
                        if (virtualIds.isNotEmpty()) {
                            val filtered = ArrayList<String>()
                            for (id in ids) {
                                if (virtualIds.contains(id)) filtered.add(id)
                            }
                            if (filtered.isNotEmpty()) {
                                param.setResult(filtered.toTypedArray())
                                XposedBridge.log("$TAG [$pkg] getCameraIdList filtered ${ids.contentToString()} -> ${filtered}")
                            }
                        }
                    }
                })
        }
    }

    private fun hookOpenCamera(cl: ClassLoader, pkg: String) {
        val hook = object : XC_MethodHook() {
            override fun beforeHookedMethod(param: MethodHookParam) {
                val id = param.args[0]
                if (id is String && preferredFrontId != null && !virtualIds.contains(id)) {
                    XposedBridge.log("$TAG [$pkg] openCamera $id -> $preferredFrontId")
                    param.args[0] = preferredFrontId
                } else {
                    XposedBridge.log("$TAG [$pkg] openCamera id=$id")
                }
            }
        }
        safeHook("openCamera(Handler)", pkg) {
            XposedHelpers.findAndHookMethod(
                "android.hardware.camera2.CameraManager", cl, "openCamera",
                String::class.java,
                "android.hardware.camera2.CameraDevice\$StateCallback",
                android.os.Handler::class.java,
                hook)
        }
    }

    private fun safeHook(what: String, pkg: String, installer: () -> Unit) {
        try {
            installer()
        } catch (t: Throwable) {
            XposedBridge.log("$TAG [$pkg] hook $what skipped: $t")
        }
    }
}
