package dev.vcam.mobile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

final class SourceRepository {
    static final class Entry {
        final String id, label;
        final File image, thumbnail, embedding;
        Entry(String id, String label, File image, File thumbnail, File embedding) {
            this.id = id; this.label = label; this.image = image;
            this.thumbnail = thumbnail; this.embedding = embedding;
        }
    }

    private final Context context;
    private final File directory, index;
    private final List<Entry> entries = new ArrayList<>();
    private String selected;

    SourceRepository(Context context) {
        this.context = context;
        directory = new File(context.getFilesDir(), "sources");
        index = new File(directory, "history.json");
        load();
    }

    List<Entry> entries() { return new ArrayList<>(entries); }
    Entry selected() {
        for (Entry entry : entries) if (entry.id.equals(selected)) return entry;
        return null;
    }

    Entry importLocal(Uri uri, FacePipeline pipeline) throws Exception {
        ImageDecoder.Source source = ImageDecoder.createSource(context.getContentResolver(), uri);
        Bitmap bitmap = ImageDecoder.decodeBitmap(source, (decoder, info, src) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
            // Detection is 640px and recognition is 112px. Capping source
            // decode avoids transiently allocating several hundred MiB for a
            // common 48/64/108 MP phone photo without discarding useful detail.
            int longest = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
            decoder.setTargetSampleSize(Math.max(1, (longest + 2047) / 2048));
        });
        Bitmap converted = bitmap.getConfig() == Bitmap.Config.ARGB_8888
                ? bitmap : bitmap.copy(Bitmap.Config.ARGB_8888, false);
        if (converted != bitmap) bitmap.recycle();
        bitmap = converted;
        Mat rgba = new Mat(), bgr = new Mat();
        Utils.bitmapToMat(bitmap, rgba);
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
        rgba.release();
        try {
            float[] embedding = pipeline.setSource(bgr);
            if (!directory.exists() && !directory.mkdirs()) {
                throw new java.io.IOException("cannot create source history");
            }
            String id = hash(bitmap);
            File image = new File(directory, id + ".jpg");
            File thumb = new File(directory, id + "-thumb.jpg");
            File latent = new File(directory, id + ".latent.f32");
            writeJpeg(bitmap, image, 94);
            int tw = 144, th = Math.max(1, bitmap.getHeight() * tw / bitmap.getWidth());
            Bitmap thumbnail = Bitmap.createScaledBitmap(bitmap, tw, th, true);
            try { writeJpeg(thumbnail, thumb, 88); }
            finally { if (thumbnail != bitmap) thumbnail.recycle(); }
            writeEmbedding(embedding, latent);
            String label = uri.getLastPathSegment() == null ? "Source" : uri.getLastPathSegment();
            entries.removeIf(item -> item.id.equals(id));
            Entry entry = new Entry(id, label, image, thumb, latent);
            entries.add(0, entry); selected = id;
            while (entries.size() > 5) {
                Entry removed = entries.remove(entries.size() - 1);
                removed.image.delete(); removed.thumbnail.delete(); removed.embedding.delete();
            }
            save();
            return entry;
        } finally {
            bitmap.recycle(); bgr.release();
        }
    }

    void select(Entry entry, FacePipeline pipeline) throws Exception {
        if (entry.embedding.isFile() && entry.embedding.length() == 512L * 4L) {
            pipeline.setSourceLatent(readEmbedding(entry.embedding));
            entries.remove(entry); entries.add(0, entry); selected = entry.id; save();
            return;
        }
        Mat bgr = read(entry.image);
        float[] embedding;
        try { embedding = pipeline.setSource(bgr); } finally { bgr.release(); }
        writeEmbedding(embedding, entry.embedding);
        entries.remove(entry); entries.add(0, entry); selected = entry.id; save();
    }

    Mat read(File image) throws Exception {
        Bitmap bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(image),
                (decoder, info, source) -> {
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                    decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
                    int longest = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                    decoder.setTargetSampleSize(Math.max(1, (longest + 2047) / 2048));
                });
        Bitmap converted = bitmap.getConfig() == Bitmap.Config.ARGB_8888
                ? bitmap : bitmap.copy(Bitmap.Config.ARGB_8888, false);
        if (converted != bitmap) bitmap.recycle();
        bitmap = converted;
        Mat rgba = new Mat(), bgr = new Mat();
        try {
            Utils.bitmapToMat(bitmap, rgba);
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            return bgr;
        } catch (Exception error) {
            bgr.release(); throw error;
        } finally {
            bitmap.recycle(); rgba.release();
        }
    }

    private void load() {
        if (!index.isFile()) return;
        try {
            JSONObject root = new JSONObject(new String(java.nio.file.Files.readAllBytes(index.toPath()), StandardCharsets.UTF_8));
            selected = root.optString("selected", null);
            JSONArray list = root.getJSONArray("entries");
            for (int i = 0; i < list.length() && entries.size() < 5; i++) {
                JSONObject item = list.getJSONObject(i);
                String id = item.getString("id");
                File image = new File(directory, id + ".jpg");
                File thumb = new File(directory, id + "-thumb.jpg");
                File embedding = new File(directory, id + ".latent.f32");
                if (image.isFile() && thumb.isFile()) entries.add(new Entry(
                        id, item.optString("label", "Source"), image, thumb, embedding));
            }
        } catch (Exception error) { android.util.Log.w("DeepLiveMobile", "Source history ignored", error); }
    }

    private void save() throws Exception {
        JSONObject root = new JSONObject(); root.put("version", 1); root.put("selected", selected);
        JSONArray list = new JSONArray();
        for (Entry entry : entries) list.put(new JSONObject().put("id", entry.id).put("label", entry.label));
        root.put("entries", list);
        File temporary = new File(index.getPath() + ".tmp");
        java.nio.file.Files.write(temporary.toPath(), (root.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
        if (!temporary.renameTo(index)) throw new java.io.IOException("cannot save source history");
    }

    private static void writeJpeg(Bitmap bitmap, File target, int quality) throws Exception {
        try (FileOutputStream output = new FileOutputStream(target)) {
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, output)) throw new java.io.IOException("JPEG encoding failed");
            output.getFD().sync();
        }
    }

    private static void writeEmbedding(float[] values, File target) throws Exception {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) bytes.putFloat(value);
        try (FileOutputStream output = new FileOutputStream(target)) {
            output.write(bytes.array()); output.getFD().sync();
        }
    }

    private static float[] readEmbedding(File source) throws Exception {
        byte[] bytes = new byte[(int) source.length()];
        try (FileInputStream input = new FileInputStream(source)) {
            int offset = 0;
            while (offset < bytes.length) {
                int count = input.read(bytes, offset, bytes.length - offset);
                if (count < 0) throw new java.io.IOException("Incomplete source embedding");
                offset += count;
            }
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[512];
        for (int i = 0; i < values.length; i++) values[i] = buffer.getFloat();
        return values;
    }

    private static String hash(Bitmap bitmap) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output);
        byte[] value = digest.digest(output.toByteArray());
        StringBuilder text = new StringBuilder();
        for (byte item : value) text.append(String.format("%02x", item & 0xff));
        return text.toString();
    }
}
