package dev.vcam.bridge

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLExt
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

internal class GlCameraRenderer(
    private val encoderSurface: Surface?,
    private val canvasWidth: Int,
    private val canvasHeight: Int,
    private val cameraWidth: Int,
    private val cameraHeight: Int,
) : AutoCloseable, SurfaceTexture.OnFrameAvailableListener {

    companion object {
        private const val TAG = "VCamRenderer"
        private const val EGL_RECORDABLE_ANDROID = 0x3142
        private const val START_TIMEOUT_SECONDS = 5L
        private const val VERTEX_SHADER = """
attribute vec2 aPosition;
attribute vec2 aTexCoord;
uniform mat4 uTextureMatrix;
uniform int uRotationQuarterTurns;
varying vec2 vTexCoord;
vec2 rotateCoord(vec2 c) {
  if (uRotationQuarterTurns == 1) return vec2(c.y, 1.0 - c.x);
  if (uRotationQuarterTurns == 2) return vec2(1.0 - c.x, 1.0 - c.y);
  if (uRotationQuarterTurns == 3) return vec2(1.0 - c.y, c.x);
  return c;
}
void main() {
  vTexCoord = rotateCoord(aTexCoord);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
"""
        private const val FRAGMENT_SHADER = """
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vTexCoord;
uniform samplerExternalOES sTexture;
void main() {
  gl_FragColor = texture2D(sTexture, vTexCoord);
}
"""
    }

    private val thread = HandlerThread("vcam-gl").also { it.start() }
    private val handler = Handler(thread.looper)
    private val ready = CountDownLatch(1)
    private val error = AtomicReference<Exception?>(null)

    private var eglDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface = EGL14.EGL_NO_SURFACE
    private var eglConfig: android.opengl.EGLConfig? = null
    private var program = 0
    private var vbo = 0
    private var textureId = 0
    private var textureMatrix = FloatArray(16)
    private var rotationQuarterTurns = 0
    private var cameraSurfaceTexture: SurfaceTexture? = null
    var cameraSurface: Surface? = null
        private set

    fun setRotationQuarterTurns(quarterTurns: Int) { rotationQuarterTurns = quarterTurns }

    fun start() {
        handler.post {
            try {
                initEgl()
                initGl()
                ready.countDown()
            } catch (e: Exception) {
                error.set(e)
                ready.countDown()
            }
        }
        try {
            if (!ready.await(START_TIMEOUT_SECONDS, TimeUnit.SECONDS)) throw IllegalStateException("EGL thread did not start")
            error.get()?.let { throw IllegalStateException("EGL init failed", it) }
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            throw IllegalStateException("EGL init interrupted", e)
        }
    }

    private fun initEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) throw RuntimeException("eglGetDisplay failed")
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) throw RuntimeException("eglInitialize failed")
        val attribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_DEPTH_SIZE, 0, EGL14.EGL_STENCIL_SIZE, 0,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1, EGL14.EGL_NONE)
        val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(eglDisplay, attribs, 0, configs, 0, 1, numConfigs, 0)) throw RuntimeException("eglChooseConfig failed")
        eglConfig = configs[0]
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
        if (eglContext == EGL14.EGL_NO_CONTEXT) throw RuntimeException("eglCreateContext failed")
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, encoderSurface, intArrayOf(EGL14.EGL_NONE), 0)
        if (eglSurface == EGL14.EGL_NO_SURFACE) throw RuntimeException("eglCreateWindowSurface failed")
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) throw RuntimeException("eglMakeCurrent failed")
    }

    private fun initGl() {
        program = createProgram()
        vbo = createVbo()
        textureId = createTexture()
        cameraSurfaceTexture = SurfaceTexture(textureId).apply { setOnFrameAvailableListener(this@GlCameraRenderer, handler) }
        cameraSurface = Surface(cameraSurfaceTexture)
    }

    private fun createProgram(): Int {
        val vertex = compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vertex)
        GLES20.glAttachShader(prog, fragment)
        GLES20.glLinkProgram(prog)
        val linkStatus = IntArray(1)
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] == 0) throw RuntimeException("Program link failed: ${GLES20.glGetProgramInfoLog(prog)}")
        return prog
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) throw RuntimeException("Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}")
        return shader
    }

    private fun createVbo(): Int {
        val quad = floatArrayOf(-1f, -1f, 0f, 1f, 1f, -1f, 1f, 1f, -1f, 1f, 0f, 0f, 1f, 1f, 1f, 0f)
        val buffer = ByteBuffer.allocateDirect(quad.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(quad).position(0)
        val ids = IntArray(1)
        GLES20.glGenBuffers(1, ids, 0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, ids[0])
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, quad.size * 4, buffer, GLES20.GL_STATIC_DRAW)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        return ids[0]
    }

    private fun createTexture(): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, ids[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
        return ids[0]
    }

    override fun onFrameAvailable(surfaceTexture: SurfaceTexture) {
        handler.post { renderFrame() }
    }

    private fun renderFrame() {
        try {
            if (eglSurface == EGL14.EGL_NO_SURFACE) return
            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) return
            cameraSurfaceTexture?.updateTexImage()
            cameraSurfaceTexture?.getTransformMatrix(textureMatrix)
            GLES20.glViewport(0, 0, canvasWidth, canvasHeight)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            GLES20.glUseProgram(program)
            val aPosition = GLES20.glGetAttribLocation(program, "aPosition")
            val aTexCoord = GLES20.glGetAttribLocation(program, "aTexCoord")
            val uTextureMatrix = GLES20.glGetUniformLocation(program, "uTextureMatrix")
            val uRotationQuarterTurns = GLES20.glGetUniformLocation(program, "uRotationQuarterTurns")
            val sTexture = GLES20.glGetUniformLocation(program, "sTexture")
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
            GLES20.glEnableVertexAttribArray(aPosition)
            GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 16, 0)
            GLES20.glEnableVertexAttribArray(aTexCoord)
            GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 16, 8)
            GLES20.glUniformMatrix4fv(uTextureMatrix, 1, false, textureMatrix, 0)
            GLES20.glUniform1i(uRotationQuarterTurns, rotationQuarterTurns)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
            GLES20.glUniform1i(sTexture, 0)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisableVertexAttribArray(aPosition)
            GLES20.glDisableVertexAttribArray(aTexCoord)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
            EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, cameraSurfaceTexture?.timestamp ?: System.nanoTime())
            if (!EGL14.eglSwapBuffers(eglDisplay, eglSurface)) Log.w(TAG, "eglSwapBuffers failed: 0x${Integer.toHexString(EGL14.eglGetError())}")
        } catch (e: Exception) {
            Log.e(TAG, "renderFrame failed", e)
        }
    }

    override fun close() {
        handler.post {
            try {
                if (program != 0) GLES20.glDeleteProgram(program)
                if (vbo != 0) { GLES20.glDeleteBuffers(1, intArrayOf(vbo), 0); vbo = 0 }
                if (textureId != 0) { GLES20.glDeleteTextures(1, intArrayOf(textureId), 0); textureId = 0 }
                cameraSurface?.release(); cameraSurface = null
                cameraSurfaceTexture?.release(); cameraSurfaceTexture = null
                if (eglSurface != EGL14.EGL_NO_SURFACE) { EGL14.eglDestroySurface(eglDisplay, eglSurface); eglSurface = EGL14.EGL_NO_SURFACE }
                if (eglContext != EGL14.EGL_NO_CONTEXT) { EGL14.eglDestroyContext(eglDisplay, eglContext); eglContext = EGL14.EGL_NO_CONTEXT }
                if (eglDisplay != EGL14.EGL_NO_DISPLAY) { EGL14.eglTerminate(eglDisplay); eglDisplay = EGL14.EGL_NO_DISPLAY }
            } catch (e: Exception) { Log.e(TAG, "close failed", e) }
        }
        thread.quitSafely()
    }
}
