package de.robv.android.xposed;

import java.lang.reflect.Member;

/** Compile-time stub. Real class is provided by the Vector framework at runtime. */
public final class XposedBridge {
    public static void log(String text) {
        throw new RuntimeException("stub");
    }

    public static void log(Throwable t) {
        throw new RuntimeException("stub");
    }

    public static XC_MethodHook.Unhook hookAllMethods(
            Class<?> hookClass, String methodName, XC_MethodHook callback) {
        throw new RuntimeException("stub");
    }

    public static XC_MethodHook.Unhook hookMethod(Member method, XC_MethodHook callback) {
        throw new RuntimeException("stub");
    }
}
