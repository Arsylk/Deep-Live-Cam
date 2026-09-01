package dev.vcam.mobile;

import android.media.Image;
import java.nio.ByteBuffer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

final class FrameTransforms {
    private static final float[] ARCFACE = {
            38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
            71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
    };

    private FrameTransforms() {}

    static Mat imageToBgr(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        Image.Plane[] planes = image.getPlanes();
        // Qualcomm (and most Camera2 implementations) expose Y plus an
        // interleaved chroma allocation through two pixel-stride-2 views. Wrap
        // those direct buffers and let OpenCV convert them without a Java
        // megapixel-by-megapixel copy.
        if (planes[0].getPixelStride() == 1
                && planes[1].getPixelStride() == 2
                && planes[2].getPixelStride() == 2) {
            Mat y = null, vu = null;
            try {
                ByteBuffer yBuffer = planes[0].getBuffer().duplicate().slice();
                ByteBuffer vuBuffer = planes[2].getBuffer().duplicate().slice();
                y = new Mat(height, width, CvType.CV_8UC1, yBuffer, planes[0].getRowStride());
                vu = new Mat(height / 2, width / 2, CvType.CV_8UC2,
                        vuBuffer, planes[2].getRowStride());
                Mat bgr = new Mat();
                Imgproc.cvtColorTwoPlane(y, vu, bgr, Imgproc.COLOR_YUV2BGR_NV21);
                return bgr;
            } catch (RuntimeException incompatibleLayout) {
                // A few devices truncate the final interleaved byte in the V
                // plane view. The portable row/pixel-stride path below remains
                // the correctness fallback.
            } finally {
                if (y != null) y.release();
                if (vu != null) vu.release();
            }
        }
        byte[] nv21 = new byte[width * height * 3 / 2];
        copyPlane(planes[0], width, height, nv21, 0, 1);
        int uvOffset = width * height;
        // Android's YUV_420_888 planes are U then V. OpenCV NV21 wants VU.
        copyChroma(planes[2], width / 2, height / 2, nv21, uvOffset);
        copyChroma(planes[1], width / 2, height / 2, nv21, uvOffset + 1);
        Mat yuv = new Mat(height + height / 2, width, CvType.CV_8UC1);
        yuv.put(0, 0, nv21);
        Mat bgr = new Mat();
        Imgproc.cvtColor(yuv, bgr, Imgproc.COLOR_YUV2BGR_NV21);
        yuv.release();
        return bgr;
    }

    private static void copyPlane(
            Image.Plane plane, int width, int height, byte[] output, int offset, int outputStep) {
        ByteBuffer buffer = plane.getBuffer().duplicate();
        int base = buffer.position();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        for (int y = 0; y < height; y++) {
            int row = y * rowStride;
            for (int x = 0; x < width; x++) {
                output[offset + (y * width + x) * outputStep] = buffer.get(base + row + x * pixelStride);
            }
        }
    }

    private static void copyChroma(
            Image.Plane plane, int width, int height, byte[] output, int offset) {
        ByteBuffer buffer = plane.getBuffer().duplicate();
        int base = buffer.position();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        int fullWidth = width * 2;
        for (int y = 0; y < height; y++) {
            int row = y * rowStride;
            for (int x = 0; x < width; x++) {
                output[offset + y * fullWidth + x * 2] = buffer.get(base + row + x * pixelStride);
            }
        }
    }

    static Mat rotateToDisplay(Mat bgr, int sensorOrientation, boolean front) {
        Mat result = new Mat();
        int normalized = ((sensorOrientation % 360) + 360) % 360;
        if (normalized == 90) Core.rotate(bgr, result, Core.ROTATE_90_CLOCKWISE);
        else if (normalized == 180) Core.rotate(bgr, result, Core.ROTATE_180);
        else if (normalized == 270) Core.rotate(bgr, result, Core.ROTATE_90_COUNTERCLOCKWISE);
        else bgr.copyTo(result);
        if (front) Core.flip(result, result, 1);
        return result;
    }

    static Mat similarity(float[] keypoints, int imageSize) {
        double ratio = imageSize % 112 == 0 ? imageSize / 112.0 : imageSize / 128.0;
        double shiftX = imageSize % 112 == 0 ? 0.0 : 8.0 * ratio;
        double sourceMeanX = 0, sourceMeanY = 0, targetMeanX = 0, targetMeanY = 0;
        double[] targetX = new double[5], targetY = new double[5];
        for (int i = 0; i < 5; i++) {
            sourceMeanX += keypoints[i * 2]; sourceMeanY += keypoints[i * 2 + 1];
            targetX[i] = ARCFACE[i * 2] * ratio + shiftX;
            targetY[i] = ARCFACE[i * 2 + 1] * ratio;
            targetMeanX += targetX[i]; targetMeanY += targetY[i];
        }
        sourceMeanX /= 5; sourceMeanY /= 5; targetMeanX /= 5; targetMeanY /= 5;
        double denominator = 0, aNumerator = 0, bNumerator = 0;
        for (int i = 0; i < 5; i++) {
            double sx = keypoints[i * 2] - sourceMeanX;
            double sy = keypoints[i * 2 + 1] - sourceMeanY;
            double tx = targetX[i] - targetMeanX;
            double ty = targetY[i] - targetMeanY;
            denominator += sx * sx + sy * sy;
            aNumerator += sx * tx + sy * ty;
            bNumerator += sx * ty - sy * tx;
        }
        if (denominator < 1e-9) return new Mat();
        double a = aNumerator / denominator, b = bNumerator / denominator;
        Mat matrix = new Mat(2, 3, CvType.CV_64FC1);
        matrix.put(0, 0,
                a, -b, targetMeanX - a * sourceMeanX + b * sourceMeanY,
                b, a, targetMeanY - b * sourceMeanX - a * sourceMeanY);
        return matrix;
    }

    static Mat align(Mat bgr, float[] keypoints, int size, Mat matrixOut) {
        Mat matrix = similarity(keypoints, size);
        if (matrix.empty()) return new Mat();
        matrix.copyTo(matrixOut);
        Mat aligned = new Mat(size, size, CvType.CV_8UC3);
        Imgproc.warpAffine(
                bgr, aligned, matrix, new Size(size, size), Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT, Scalar.all(0));
        matrix.release();
        return aligned;
    }

    static Mat alignedWithBorder(Mat bgr, Mat matrix, int size, int borderMode) {
        Mat aligned = new Mat(size, size, CvType.CV_8UC3);
        Imgproc.warpAffine(bgr, aligned, matrix, new Size(size, size), Imgproc.INTER_LINEAR,
                borderMode, Scalar.all(0));
        return aligned;
    }

    static float[] nchwRgb(Mat bgr, float mean, float scale) {
        int height = bgr.rows();
        int width = bgr.cols();
        byte[] bytes = new byte[width * height * 3];
        bgr.get(0, 0, bytes);
        int plane = width * height;
        float[] out = new float[plane * 3];
        for (int i = 0; i < plane; i++) {
            int base = i * 3;
            out[i] = ((bytes[base + 2] & 0xff) - mean) / scale;
            out[plane + i] = ((bytes[base + 1] & 0xff) - mean) / scale;
            out[plane * 2 + i] = ((bytes[base] & 0xff) - mean) / scale;
        }
        return out;
    }

    static Mat rgbOutputToBgr(float[] output, int size) {
        int plane = size * size;
        byte[] bgr = new byte[plane * 3];
        for (int i = 0; i < plane; i++) {
            bgr[i * 3] = toByte(output[plane * 2 + i]);
            bgr[i * 3 + 1] = toByte(output[plane + i]);
            bgr[i * 3 + 2] = toByte(output[i]);
        }
        Mat result = new Mat(size, size, CvType.CV_8UC3);
        result.put(0, 0, bgr);
        return result;
    }

    private static byte toByte(float normalized) {
        return (byte) Math.max(0, Math.min(255, (int) (normalized * 255.0f)));
    }

    static Mat softMask(int size) {
        Mat mask = Mat.zeros(size, size, CvType.CV_8UC1);
        Imgproc.ellipse(mask, new Point(size / 2.0, size / 2.0),
                new Size(size * 0.44, size * 0.44), 0, 0, 360, Scalar.all(255), -1);
        Imgproc.GaussianBlur(mask, mask, new Size(31, 31), 12);
        return mask;
    }

    static Mat paste(Mat original, Mat generated, Mat affine, Mat mask) {
        Mat inverse = new Mat();
        Imgproc.invertAffineTransform(affine, inverse);
        Mat warped = new Mat(original.size(), CvType.CV_8UC3);
        Mat alpha = new Mat(original.size(), CvType.CV_8UC1);
        Imgproc.warpAffine(generated, warped, inverse, original.size(), Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE, Scalar.all(0));
        Imgproc.warpAffine(mask, alpha, inverse, original.size(), Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT, Scalar.all(0));
        Mat alpha3 = new Mat();
        Imgproc.cvtColor(alpha, alpha3, Imgproc.COLOR_GRAY2BGR);
        Mat inverseAlpha = new Mat(), fullAlpha = new Mat(alpha3.size(), alpha3.type(), Scalar.all(255));
        Core.subtract(fullAlpha, alpha3, inverseAlpha);
        Mat left = new Mat();
        Mat right = new Mat();
        Core.multiply(warped, alpha3, left, 1.0 / 255.0);
        Core.multiply(original, inverseAlpha, right, 1.0 / 255.0);
        Mat result = new Mat();
        Core.add(left, right, result);
        inverse.release(); warped.release(); alpha.release(); alpha3.release(); fullAlpha.release();
        inverseAlpha.release(); left.release(); right.release();
        return result;
    }

    static Mat colorMatch(Mat generated, Mat target, Mat alpha, float strength,
                          double[] priorScale, double[] priorOffset) {
        Mat mask = new Mat();
        Imgproc.threshold(alpha, mask, 96, 255, Imgproc.THRESH_BINARY);
        Mat generatedLab = new Mat(), targetLab = new Mat();
        Imgproc.cvtColor(generated, generatedLab, Imgproc.COLOR_BGR2Lab);
        Imgproc.cvtColor(target, targetLab, Imgproc.COLOR_BGR2Lab);
        MatOfDouble meanG = new MatOfDouble(), stdG = new MatOfDouble();
        MatOfDouble meanT = new MatOfDouble(), stdT = new MatOfDouble();
        Core.meanStdDev(generatedLab, meanG, stdG, mask);
        Core.meanStdDev(targetLab, meanT, stdT, mask);
        double[] meansG = meanG.toArray(), stdsG = stdG.toArray();
        double[] meansT = meanT.toArray(), stdsT = stdT.toArray();
        double[] scale = new double[3], offset = new double[3];
        for (int c = 0; c < 3; c++) {
            double gs = stdsG[c] + .001, ts = stdsT[c] + .001;
            scale[c] = Math.max(.85, Math.min(1.18, ts / gs));
            offset[c] = Math.max(-12, Math.min(12,
                    meansT[c] - meansG[c] * scale[c]));
            if (!Double.isNaN(priorScale[c])) {
                scale[c] = priorScale[c] * .82 + scale[c] * .18;
                offset[c] = priorOffset[c] * .82 + offset[c] * .18;
            }
            priorScale[c] = scale[c]; priorOffset[c] = offset[c];
        }
        java.util.List<Mat> channels = new java.util.ArrayList<>();
        Core.split(generatedLab, channels);
        for (int c = 0; c < 3; c++) {
            double effectiveScale = 1.0 + (scale[c] - 1.0) * strength;
            double effectiveOffset = offset[c] * strength;
            channels.get(c).convertTo(channels.get(c), CvType.CV_8UC1, effectiveScale, effectiveOffset);
        }
        Mat matchedLab = new Mat(), matched = new Mat();
        Core.merge(channels, matchedLab);
        Imgproc.cvtColor(matchedLab, matched, Imgproc.COLOR_Lab2BGR);
        for (Mat channel : channels) channel.release();
        mask.release(); generatedLab.release(); targetLab.release(); meanG.release(); stdG.release();
        meanT.release(); stdT.release(); matchedLab.release();
        return matched;
    }

    static void sharpenFace(Mat frame, float[] bbox, float strength) {
        if (strength <= 0) return;
        int x1 = Math.max(0, Math.min(frame.cols() - 1, (int) bbox[0]));
        int y1 = Math.max(0, Math.min(frame.rows() - 1, (int) bbox[1]));
        int x2 = Math.max(x1 + 1, Math.min(frame.cols(), (int) bbox[2]));
        int y2 = Math.max(y1 + 1, Math.min(frame.rows(), (int) bbox[3]));
        Mat roi = frame.submat(y1, y2, x1, x2), blurred = new Mat(), sharpened = new Mat();
        Imgproc.GaussianBlur(roi, blurred, new Size(0, 0), 3);
        Core.addWeighted(roi, 1.0 + strength, blurred, -strength, 0, sharpened);
        Mat localMask = Mat.zeros(roi.rows(), roi.cols(), CvType.CV_8UC1);
        Imgproc.ellipse(localMask, new Point(roi.cols() / 2.0, roi.rows() / 2.0),
                new Size(Math.max(1, roi.cols() / 2.0 - 1), Math.max(1, roi.rows() / 2.0 - 1)),
                0, 0, 360, Scalar.all(255), -1);
        int kernel = Math.max(3, Math.min(31, Math.round(Math.min(roi.rows(), roi.cols()) * .08f) | 1));
        Imgproc.GaussianBlur(localMask, localMask, new Size(kernel, kernel), 0);
        Mat alpha3 = new Mat(); Imgproc.cvtColor(localMask, alpha3, Imgproc.COLOR_GRAY2BGR);
        Mat inverse = new Mat(), full = new Mat(alpha3.size(), alpha3.type(), Scalar.all(255));
        Core.subtract(full, alpha3, inverse);
        Mat left = new Mat(), right = new Mat(), blended = new Mat();
        Core.multiply(sharpened, alpha3, left, 1.0 / 255.0);
        Core.multiply(roi, inverse, right, 1.0 / 255.0);
        Core.add(left, right, blended); blended.copyTo(roi);
        roi.release(); blurred.release(); sharpened.release(); localMask.release(); alpha3.release();
        inverse.release(); full.release(); left.release(); right.release(); blended.release();
    }
}
