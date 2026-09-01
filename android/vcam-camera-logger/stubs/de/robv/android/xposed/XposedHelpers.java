package de.robv.android.xposed;

/** Compile-time stub. Real class is provided by the Vector framework at runtime. */
public final class XposedHelpers {
    public static Class<?> findClass(String className, ClassLoader classLoader) {
        throw new RuntimeException("stub");
    }

    /** Canonical helper: hooks every method with the given name and params. */
    public static XC_MethodHook.Unhook findAndHookMethod(
            String className, ClassLoader classLoader, String methodName,
            Object... parameterTypesAndCallback) {
        throw new RuntimeException("stub");
    }

    public static XC_MethodHook.Unhook findAndHookMethod(
            Class<?> clazz, String methodName, Object... parameterTypesAndCallback) {
        throw new RuntimeException("stub");
    }

    public static Object callMethod(Object obj, String methodName, Object... args) {
        throw new RuntimeException("stub");
    }

    public static Object callStaticMethod(Class<?> clazz, String methodName,
            Object... args) {
        throw new RuntimeException("stub");
    }

    public static Object getObjectField(Object obj, String fieldName) {
        throw new RuntimeException("stub");
    }
}
