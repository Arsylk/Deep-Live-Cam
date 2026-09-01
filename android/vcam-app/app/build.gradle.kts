import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.vcam.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.vcam.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 7
        versionName = "1.7-fit-native-aspect"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
    implementation(kotlin("stdlib"))
}
