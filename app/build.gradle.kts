plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "cn.yege.dshcam"
    compileSdk = 35

    defaultConfig {
        applicationId = "cn.yege.dshcam"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            // v1 不做混淆/压缩，避免额外的 proguard 文件依赖
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    val cameraxVersion = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    implementation("androidx.camera:camera-mlkit-vision:$cameraxVersion")

    // v1 只用 face-detection；image-labeling 先引入，v1.1 场景标签再用
    implementation("com.google.mlkit:face-detection:16.1.7")
    implementation("com.google.mlkit:image-labeling:17.0.9")

    // Rules.kt 是纯 Kotlin（不 import android.*），可在普通 JVM 上跑单元测试
    testImplementation("junit:junit:4.13.2")

    // ★ 2026-10-04：1:1 画幅要在拍完后居中裁剪，重写文件时得把原图 EXIF 方向带回去
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // ★ 2026-10-05：主体模型（u2netp 显著目标检测）跑 ONNX。
    //   规则算法（AutoFrame.salienceFromArgb）分不出"深色主体 + 大片渐晕背景"，
    //   模型 4.4 MB 打进 assets，直接用 ONNX Runtime 跑，免转换。
    //   AAR 约 26.6 MB（含 4 个 ABI；只打 arm64 时约 7 MB）。
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    // 注：实时调色（LutEffect.kt）走 CameraEffect + 自写 SurfaceProcessor，
    //     GL 层自己用 EGL14/GLES20 写（camera-effects 里的 opengl.* 全是
    //     @RestrictTo(LIBRARY) 或包级私有，app 侧用不了），所以不需要额外依赖。
}
