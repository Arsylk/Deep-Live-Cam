package dev.vcam.bridge;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLExt;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Surface;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Zero-copy Camera2-to-MediaCodec renderer with a configurable rotation.
 *
 * <p>Camera2 produces into an external-OES texture.  This class renders that
 * texture into the MediaCodec input surface on a dedicated EGL thread.  A
 * 90/270 degree portrait image is fitted inside the fixed 1280x720 transport
 * canvas, preserving aspect ratio and using black side bars instead of
 * stretching or silently cropping the camera.</p>
 */
final class GlCameraRenderer implements AutoCloseable,
        SurfaceTexture.OnFrameAvailableListener {
    private static final String TAG = "VCamRenderer";
    private static final int EGL_RECORDABLE_ANDROID = 0x3142;
    private static final long START_TIMEOUT_SECONDS = 5;

    private static final String VERTEX_SHADER =
            "attribute vec2 aPosition;\n" +
            "attribute vec2 aTexCoord;\n" +
            "uniform mat4 uTextureMatrix;\n" +
            "uniform int uRotationQuarterTurns;\n" +
            "varying vec2 vTexCoord;\n" +
            "vec2 rotateCoord(vec2 c) {\n" +
            "  if (uRotationQuarterTurns == 1) return vec2(c.y, 1.0 - c.x);\n" +
            "  if (uRotationQuarterTurns == 2) return vec2(1.0 - c.x, 1.0 - c.y);\n" +
            "  if (uRotationQuarterTurns == 3) return vec2(1.0 - c.y, c.x);\n" +
            "  return c;\n" +
            "}\n" +
            "void main() {\n" +
            "  gl_Position = vec4(aPosition, 0.0, 1.0);\n" +
            "  vec2 rotated = rotateCoord(aTexCoord);\n" +
            "  vTexCoord = (uTextureMatrix * vec4(rotated, 0.0, 1.0)).xy;\n" +
            "}\n";

    private static final String FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;\n" +
            "uniform samplerExternalOES uTexture;\n" +
            "varying vec2 vTexCoord;\n" +
            "void main() {\n" +
            "  gl_FragColor = texture2D(uTexture, vTexCoord);\n" +
            "}\n";

    private static final float[] TEXTURE_COORDINATES = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
    };

    private final Surface encoderSurface;
    private final int inputWidth;
    private final int inputHeight;
    private final int outputWidth;
    private final int outputHeight;
    private final AtomicReference<Throwable> asynchronousError = new AtomicReference<>();

    private HandlerThread renderThread;
    private Handler renderHandler;
    private android.opengl.EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private android.opengl.EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private android.opengl.EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private SurfaceTexture cameraTexture;
    private Surface cameraSurface;
    private int textureId;
    private int program;
    private int positionLocation;
    private int textureCoordinateLocation;
    private int textureMatrixLocation;
    private int rotationLocation;
    private FloatBuffer positionBuffer;
    private FloatBuffer textureCoordinateBuffer;
    private final float[] textureMatrix = new float[16];
    private volatile int rotationDegrees;
    private volatile int textureMatrixRotationDegrees;
    private volatile int shaderRotationDegrees;
    private volatile boolean closing;
    private long renderedFrames;

    GlCameraRenderer(
            Surface encoderSurface,
            int inputWidth,
            int inputHeight,
            int outputWidth,
            int outputHeight) {
        this.encoderSurface = encoderSurface;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
    }

    Surface start() throws IOException {
        if (renderThread != null) {
            return cameraSurface;
        }
        CountDownLatch ready = new CountDownLatch(1);
        AtomicReference<Throwable> startupError = new AtomicReference<>();
        renderThread = new HandlerThread("vcam-gl-renderer");
        renderThread.start();
        renderHandler = new Handler(renderThread.getLooper());
        renderHandler.post(() -> {
            try {
                initializeGl();
            } catch (Throwable error) {
                startupError.set(error);
            } finally {
                ready.countDown();
            }
        });
        try {
            if (!ready.await(START_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                throw new IOException("timed out initializing the camera rotation renderer");
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while initializing the camera renderer", error);
        }
        if (startupError.get() != null) {
            close();
            throw new IOException("camera rotation renderer failed", startupError.get());
        }
        return cameraSurface;
    }

    Surface getCameraSurface() {
        Throwable error = asynchronousError.get();
        if (error != null) {
            throw new IllegalStateException("camera renderer stopped", error);
        }
        if (cameraSurface == null) {
            throw new IllegalStateException("camera renderer is not initialized");
        }
        return cameraSurface;
    }

    void setRotationDegrees(int degrees) {
        int normalized = ((degrees % 360) + 360) % 360;
        if (normalized % 90 != 0) {
            throw new IllegalArgumentException("rotation must be a multiple of 90 degrees");
        }
        rotationDegrees = normalized;
    }

    int getRotationDegrees() {
        return rotationDegrees;
    }

    long getRenderedFrames() {
        return renderedFrames;
    }

    int getTextureMatrixRotationDegrees() {
        return textureMatrixRotationDegrees;
    }

    int getShaderRotationDegrees() {
        return shaderRotationDegrees;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture ignored) {
        if (closing || renderHandler == null) {
            return;
        }
        // The listener already runs on renderHandler. Rendering immediately
        // keeps one latest SurfaceTexture frame and avoids another queue.
        try {
            renderFrame();
        } catch (Throwable error) {
            if (asynchronousError.compareAndSet(null, error)) {
                Log.e(TAG, "Camera frame rendering failed", error);
            }
        }
    }

    private void initializeGl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new IllegalStateException("eglGetDisplay failed");
        }
        int[] version = new int[2];
        requireEgl(EGL14.eglInitialize(eglDisplay, version, 0, version, 1), "eglInitialize");
        int[] configurationAttributes = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
                EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE,
        };
        android.opengl.EGLConfig[] configurations = new android.opengl.EGLConfig[1];
        int[] configurationCount = new int[1];
        requireEgl(
                EGL14.eglChooseConfig(
                        eglDisplay,
                        configurationAttributes,
                        0,
                        configurations,
                        0,
                        configurations.length,
                        configurationCount,
                        0) && configurationCount[0] > 0,
                "eglChooseConfig");
        int[] contextAttributes = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE,
        };
        eglContext = EGL14.eglCreateContext(
                eglDisplay,
                configurations[0],
                EGL14.EGL_NO_CONTEXT,
                contextAttributes,
                0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw new IllegalStateException("eglCreateContext failed: 0x" +
                    Integer.toHexString(EGL14.eglGetError()));
        }
        eglSurface = EGL14.eglCreateWindowSurface(
                eglDisplay,
                configurations[0],
                encoderSurface,
                new int[] {EGL14.EGL_NONE},
                0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new IllegalStateException("eglCreateWindowSurface failed: 0x" +
                    Integer.toHexString(EGL14.eglGetError()));
        }
        requireEgl(
                EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext),
                "eglMakeCurrent");

        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        positionLocation = GLES20.glGetAttribLocation(program, "aPosition");
        textureCoordinateLocation = GLES20.glGetAttribLocation(program, "aTexCoord");
        textureMatrixLocation = GLES20.glGetUniformLocation(program, "uTextureMatrix");
        rotationLocation = GLES20.glGetUniformLocation(program, "uRotationQuarterTurns");
        checkGl("shader locations");

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S,
                GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T,
                GLES20.GL_CLAMP_TO_EDGE);
        checkGl("external texture setup");

        positionBuffer = directFloatBuffer(new float[8]);
        textureCoordinateBuffer = directFloatBuffer(TEXTURE_COORDINATES);
        cameraTexture = new SurfaceTexture(textureId);
        cameraTexture.setDefaultBufferSize(inputWidth, inputHeight);
        cameraTexture.setOnFrameAvailableListener(this, renderHandler);
        cameraSurface = new Surface(cameraTexture);
        Log.i(
                TAG,
                "OpenGL ES rotation renderer ready: input=" + inputWidth + "x" +
                        inputHeight + " output=" + outputWidth + "x" + outputHeight);
    }

    private void renderFrame() {
        if (cameraTexture == null || eglDisplay == EGL14.EGL_NO_DISPLAY) {
            return;
        }
        requireEgl(
                EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext),
                "eglMakeCurrent(frame)");
        cameraTexture.updateTexImage();
        cameraTexture.getTransformMatrix(textureMatrix);
        int desiredRotation = rotationDegrees;
        int matrixQuarterTurns = detectTextureQuarterTurns(textureMatrix);
        int shaderQuarterTurns =
                (desiredRotation / 90 - matrixQuarterTurns + 4) % 4;
        textureMatrixRotationDegrees = matrixQuarterTurns * 90;
        shaderRotationDegrees = shaderQuarterTurns * 90;
        updatePositionBuffer(desiredRotation);

        GLES20.glViewport(0, 0, outputWidth, outputHeight);
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        positionBuffer.position(0);
        GLES20.glEnableVertexAttribArray(positionLocation);
        GLES20.glVertexAttribPointer(
                positionLocation, 2, GLES20.GL_FLOAT, false, 0, positionBuffer);
        textureCoordinateBuffer.position(0);
        GLES20.glEnableVertexAttribArray(textureCoordinateLocation);
        GLES20.glVertexAttribPointer(
                textureCoordinateLocation,
                2,
                GLES20.GL_FLOAT,
                false,
                0,
                textureCoordinateBuffer);
        GLES20.glUniformMatrix4fv(textureMatrixLocation, 1, false, textureMatrix, 0);
        GLES20.glUniform1i(rotationLocation, shaderQuarterTurns);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        checkGl("draw camera frame");

        long timestamp = cameraTexture.getTimestamp();
        if (timestamp > 0) {
            EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, timestamp);
        }
        requireEgl(EGL14.eglSwapBuffers(eglDisplay, eglSurface), "eglSwapBuffers");
        renderedFrames++;
        if (renderedFrames == 1) {
            Log.i(
                    TAG,
                    String.format(
                            Locale.US,
                            "First frame desiredRotation=%d textureRotation=%d " +
                                    "shaderRotation=%d textureMatrix=%s",
                            desiredRotation,
                            textureMatrixRotationDegrees,
                            shaderRotationDegrees,
                            Arrays.toString(textureMatrix)));
        }
    }

    /**
     * Extract the producer's nearest quarter-turn from SurfaceTexture's
     * column-major affine matrix. Crop scales and a vertical GL-origin flip
     * do not change which input axis dominates each output axis.
     */
    private static int detectTextureQuarterTurns(float[] matrix) {
        float xFromU = matrix[0];
        float yFromU = matrix[1];
        float xFromV = matrix[4];
        float yFromV = matrix[5];
        float diagonal = Math.abs(xFromU) + Math.abs(yFromV);
        float offDiagonal = Math.abs(yFromU) + Math.abs(xFromV);
        if (offDiagonal > diagonal) {
            return xFromV >= 0.0f && yFromU <= 0.0f ? 1 : 3;
        }
        return xFromU >= 0.0f ? 0 : 2;
    }

    private void updatePositionBuffer(int degrees) {
        boolean quarterTurn = degrees == 90 || degrees == 270;
        float contentAspect = quarterTurn
                ? (float) inputHeight / (float) inputWidth
                : (float) inputWidth / (float) inputHeight;
        float outputAspect = (float) outputWidth / (float) outputHeight;
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        if (contentAspect > outputAspect) {
            scaleY = outputAspect / contentAspect;
        } else {
            scaleX = contentAspect / outputAspect;
        }
        float[] positions = {
                -scaleX, -scaleY,
                scaleX, -scaleY,
                -scaleX, scaleY,
                scaleX, scaleY,
        };
        positionBuffer.position(0);
        positionBuffer.put(positions);
        positionBuffer.position(0);
    }

    private static FloatBuffer directFloatBuffer(float[] values) {
        FloatBuffer buffer = ByteBuffer
                .allocateDirect(values.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        buffer.put(values);
        buffer.position(0);
        return buffer;
    }

    private static int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragmentShader = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        int result = GLES20.glCreateProgram();
        GLES20.glAttachShader(result, vertexShader);
        GLES20.glAttachShader(result, fragmentShader);
        GLES20.glLinkProgram(result);
        int[] linked = new int[1];
        GLES20.glGetProgramiv(result, GLES20.GL_LINK_STATUS, linked, 0);
        String message = GLES20.glGetProgramInfoLog(result);
        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        if (linked[0] == 0) {
            GLES20.glDeleteProgram(result);
            throw new IllegalStateException("OpenGL program link failed: " + message);
        }
        return result;
    }

    private static int compileShader(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String message = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new IllegalStateException("OpenGL shader compilation failed: " + message);
        }
        return shader;
    }

    private static void checkGl(String operation) {
        int error = GLES20.glGetError();
        if (error != GLES20.GL_NO_ERROR) {
            throw new IllegalStateException(
                    operation + " failed with GL error 0x" + Integer.toHexString(error));
        }
    }

    private static void requireEgl(boolean result, String operation) {
        if (!result) {
            throw new IllegalStateException(
                    operation + " failed with EGL error 0x" +
                            Integer.toHexString(EGL14.eglGetError()));
        }
    }

    @Override
    public void close() {
        closing = true;
        Handler handler = renderHandler;
        HandlerThread thread = renderThread;
        if (handler != null && thread != null) {
            CountDownLatch released = new CountDownLatch(1);
            handler.post(() -> {
                try {
                    releaseGl();
                } finally {
                    released.countDown();
                }
            });
            try {
                released.await(2, TimeUnit.SECONDS);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
            }
            thread.quitSafely();
            try {
                thread.join(2000);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
            }
        } else {
            releaseGl();
        }
        renderHandler = null;
        renderThread = null;
    }

    private void releaseGl() {
        if (cameraTexture != null) {
            cameraTexture.setOnFrameAvailableListener(null);
        }
        if (cameraSurface != null) {
            cameraSurface.release();
            cameraSurface = null;
        }
        if (cameraTexture != null) {
            cameraTexture.release();
            cameraTexture = null;
        }
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(
                    eglDisplay,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT);
            if (program != 0) {
                GLES20.glDeleteProgram(program);
                program = 0;
            }
            if (textureId != 0) {
                GLES20.glDeleteTextures(1, new int[] {textureId}, 0);
                textureId = 0;
            }
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
                eglSurface = EGL14.EGL_NO_SURFACE;
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
                eglContext = EGL14.EGL_NO_CONTEXT;
            }
            EGL14.eglReleaseThread();
            EGL14.eglTerminate(eglDisplay);
            eglDisplay = EGL14.EGL_NO_DISPLAY;
        }
    }
}
