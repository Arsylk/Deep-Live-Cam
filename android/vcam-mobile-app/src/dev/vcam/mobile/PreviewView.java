package dev.vcam.mobile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

final class PreviewView extends View {
    private final Object frameLock = new Object();
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Mat rgba = new Mat();
    private Bitmap current;

    PreviewView(Context context) { super(context); setBackgroundColor(Color.BLACK); }
    PreviewView(Context context, AttributeSet attrs) { super(context, attrs); setBackgroundColor(Color.BLACK); }

    void show(Mat bgr) {
        synchronized (frameLock) {
            org.opencv.imgproc.Imgproc.cvtColor(bgr, rgba, org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA);
            if (current == null || current.getWidth() != rgba.cols() || current.getHeight() != rgba.rows()) {
                if (current != null && !current.isRecycled()) current.recycle();
                current = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
            }
            Utils.matToBitmap(rgba, current);
        }
        postInvalidateOnAnimation();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        synchronized (frameLock) {
            if (current == null) return;
            float scale = Math.min(getWidth() / (float) current.getWidth(), getHeight() / (float) current.getHeight());
            float left = (getWidth() - current.getWidth() * scale) / 2f;
            float top = (getHeight() - current.getHeight() * scale) / 2f;
            canvas.save(); canvas.translate(left, top); canvas.scale(scale, scale);
            canvas.drawBitmap(current, 0, 0, paint); canvas.restore();
        }
    }

    void release() {
        synchronized (frameLock) {
            rgba.release();
            if (current != null && !current.isRecycled()) current.recycle();
            current = null;
        }
    }
}
