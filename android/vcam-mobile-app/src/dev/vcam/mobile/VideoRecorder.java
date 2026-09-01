package dev.vcam.mobile;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.os.Environment;
import java.io.File;
import java.nio.ByteBuffer;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

final class VideoRecorder implements AutoCloseable {
    private MediaCodec codec;
    private MediaMuxer muxer;
    private int track = -1;
    private boolean muxerStarted;
    private long firstTimestampNs, lastPtsUs;
    private File output;
    private int width, height;

    synchronized boolean active() { return codec != null; }
    synchronized File output() { return output; }

    synchronized void start(File moviesDirectory, int width, int height) throws Exception {
        close();
        try {
            this.width = width & ~1; this.height = height & ~1;
            if (!moviesDirectory.exists() && !moviesDirectory.mkdirs()) throw new java.io.IOException("cannot create recordings directory");
            output = new File(moviesDirectory, "deep-live-" + new java.text.SimpleDateFormat(
                    "yyyyMMdd-HHmmss", java.util.Locale.US).format(new java.util.Date()) + ".mp4");
            MediaFormat format = MediaFormat.createVideoFormat("video/avc", this.width, this.height);
            // OpenCV produces I420 below. Request planar YUV explicitly; Flexible
            // lets Qualcomm choose NV12 and silently swaps/interleaves our chroma.
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar);
            format.setInteger(MediaFormat.KEY_BIT_RATE, 5_000_000);
            format.setInteger(MediaFormat.KEY_FRAME_RATE, 30);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
            String encoder = new MediaCodecList(MediaCodecList.REGULAR_CODECS).findEncoderForFormat(format);
            if (encoder == null) throw new java.io.IOException("no planar YUV H.264 encoder");
            codec = MediaCodec.createByCodecName(encoder);
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            muxer = new MediaMuxer(output.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            track = -1; muxerStarted = false; firstTimestampNs = 0; lastPtsUs = -1;
        } catch (Exception error) {
            File failed = output;
            close();
            if (failed != null && failed.isFile()) failed.delete();
            output = null;
            throw error;
        }
    }

    synchronized boolean frame(Mat bgr, long timestampNs) {
        if (codec == null) return false;
        try {
            Mat sized = bgr;
            Mat i420 = null;
            byte[] bytes;
            try {
                if (bgr.cols() != width || bgr.rows() != height) {
                    sized = new Mat(); Imgproc.resize(bgr, sized, new org.opencv.core.Size(width, height));
                }
                i420 = new Mat(); Imgproc.cvtColor(sized, i420, Imgproc.COLOR_BGR2YUV_I420);
                bytes = new byte[(int) (i420.total() * i420.elemSize())]; i420.get(0, 0, bytes);
            } finally {
                if (i420 != null) i420.release();
                if (sized != bgr) sized.release();
            }
            int index = codec.dequeueInputBuffer(0);
            if (index >= 0) {
                ByteBuffer input = codec.getInputBuffer(index); input.clear();
                if (input.remaining() < bytes.length) throw new java.io.IOException(
                        "encoder input is smaller than the I420 frame");
                input.put(bytes);
                if (firstTimestampNs == 0) firstTimestampNs = timestampNs;
                long pts = Math.max(lastPtsUs + 1, (timestampNs - firstTimestampNs) / 1000);
                lastPtsUs = pts;
                codec.queueInputBuffer(index, 0, bytes.length, pts, 0);
            }
            drain(false);
            return true;
        } catch (Exception error) {
            android.util.Log.e("DeepLiveMobile", "Recording frame failed; stopping recorder", error);
            close();
            return false;
        }
    }

    private void drain(boolean end) throws Exception {
        if (codec == null) return;
        long deadline = android.os.SystemClock.elapsedRealtime() + 2000;
        if (end) {
            int input;
            do { input = codec.dequeueInputBuffer(10_000); }
            while (input < 0 && android.os.SystemClock.elapsedRealtime() < deadline);
            if (input >= 0) codec.queueInputBuffer(input, 0, 0, Math.max(0, lastPtsUs + 1), MediaCodec.BUFFER_FLAG_END_OF_STREAM);
        }
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (true) {
            int outputIndex = codec.dequeueOutputBuffer(info, end ? 10_000 : 0);
            if (outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!end || android.os.SystemClock.elapsedRealtime() >= deadline) break;
                continue;
            }
            if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                track = muxer.addTrack(codec.getOutputFormat()); muxer.start(); muxerStarted = true; continue;
            }
            if (outputIndex < 0) continue;
            ByteBuffer buffer = codec.getOutputBuffer(outputIndex);
            if (info.size > 0 && muxerStarted) {
                buffer.position(info.offset); buffer.limit(info.offset + info.size);
                muxer.writeSampleData(track, buffer, info);
            }
            boolean eos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            codec.releaseOutputBuffer(outputIndex, false);
            if (eos || !end) break;
        }
    }

    @Override public synchronized void close() {
        if (codec != null) {
            try { drain(true); } catch (Exception ignored) {}
            try { codec.stop(); } catch (Exception ignored) {}
            codec.release(); codec = null;
        }
        if (muxer != null) {
            try { if (muxerStarted) muxer.stop(); } catch (Exception ignored) {}
            muxer.release(); muxer = null;
        }
        muxerStarted = false;
    }
}
