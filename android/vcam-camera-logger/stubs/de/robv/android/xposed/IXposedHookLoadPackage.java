package de.robv.android.xposed;

import de.robv.android.xposed.callbacks.XC_LoadPackage;

/** Compile-time stub. Real class is provided by the Vector framework at runtime. */
public interface IXposedHookLoadPackage {
    void handleLoadPackage(XC_LoadPackage.LoadPackageParam lpparam) throws Throwable;
}
