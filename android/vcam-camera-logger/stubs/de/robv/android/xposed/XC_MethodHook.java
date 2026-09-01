package de.robv.android.xposed;

import java.lang.reflect.Member;

/** Compile-time stub. Real class is provided by the Vector framework at runtime. */
public abstract class XC_MethodHook {
    protected void beforeHookedMethod(MethodHookParam param) throws Throwable {}

    protected void afterHookedMethod(MethodHookParam param) throws Throwable {}

    public static class MethodHookParam {
        public Member method;
        public Object thisObject;
        public Object[] args;
        public Object result;

        public Object getResult() {
            return result;
        }

        public void setResult(Object result) {
            this.result = result;
        }
    }

    public static class Unhook {}
}
