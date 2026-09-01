package dev.vcam.mobile;

import android.content.Context;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import org.json.JSONObject;

final class ModelStore {
    static final String DETECTOR = "det_10g_640.onnx";
    static final String RECOGNIZER = "w600k_r50_112.onnx";
    static final String SWAPPER = "inswapper_128_mobile.onnx";
    static final String SWAPPER_FP16 = "inswapper_128_mobile_fp16.onnx";
    static final String EMAP = "inswapper_emap.f32";
    static final String QNN_PROBE = "qnn_probe.onnx";
    private final Context context;
    private final File directory;

    ModelStore(Context context) {
        this.context = context.getApplicationContext();
        directory = new File(context.getFilesDir(), "models");
    }

    File file(String name) { return new File(directory, name); }

    boolean ready() {
        return file(DETECTOR).isFile() && file(RECOGNIZER).isFile()
                && file(SWAPPER).isFile() && file(SWAPPER_FP16).isFile()
                && file(EMAP).isFile()
                && file(QNN_PROBE).isFile();
    }

    String readiness() {
        if (ready()) return "Offline model pack ready";
        return "Model pack missing. Run install.sh once over ADB; the app itself never downloads models.";
    }

    void verify() throws Exception {
        File manifestFile = file("manifest.json");
        if (!manifestFile.isFile()) throw new IOException("manifest.json is missing");
        String text;
        try (FileInputStream input = new FileInputStream(manifestFile)) {
            text = new String(readAll(input), java.nio.charset.StandardCharsets.UTF_8);
        }
        JSONObject models = new JSONObject(text).getJSONObject("models");
        StringBuilder fingerprintSource = new StringBuilder(sha256(manifestFile));
        for (String name : new String[]{
                DETECTOR, RECOGNIZER, SWAPPER, SWAPPER_FP16, EMAP, QNN_PROBE}) {
            File model = file(name);
            JSONObject expected = models.getJSONObject(name);
            if (!model.isFile() || model.length() != expected.getLong("bytes")) {
                throw new IOException(name + " is missing or incomplete");
            }
            fingerprintSource.append('|').append(name).append(':')
                    .append(model.length()).append(':').append(model.lastModified());
        }
        String fingerprint = sha256(fingerprintSource.toString().getBytes(
                java.nio.charset.StandardCharsets.UTF_8));
        android.content.SharedPreferences preferences = context.getSharedPreferences(
                "verified-model-pack", Context.MODE_PRIVATE);
        if (fingerprint.equals(preferences.getString("fingerprint", null))) return;
        for (String name : new String[]{
                DETECTOR, RECOGNIZER, SWAPPER, SWAPPER_FP16, EMAP, QNN_PROBE}) {
            File model = file(name);
            JSONObject expected = models.getJSONObject(name);
            String actual = sha256(model);
            if (!actual.equalsIgnoreCase(expected.getString("sha256"))) {
                throw new IOException(name + " checksum mismatch");
            }
        }
        preferences.edit().putString("fingerprint", fingerprint).apply();
    }

    static byte[] readAll(InputStream input) throws IOException {
        byte[] buffer = new byte[65536];
        java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
        int count;
        while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
        return output.toByteArray();
    }

    static void copy(InputStream input, File target) throws IOException {
        File parent = target.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("cannot create " + parent);
        }
        File temporary = new File(target.getPath() + ".tmp");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
            output.getFD().sync();
        }
        if (!temporary.renameTo(target)) throw new IOException("cannot install " + target);
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (FileInputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) digest.update(buffer, 0, count);
        }
        return hex(digest.digest());
    }

    private static String sha256(byte[] bytes) throws Exception {
        return hex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    private static String hex(byte[] bytes) {
        StringBuilder value = new StringBuilder();
        for (byte item : bytes) value.append(String.format("%02x", item & 0xff));
        return value.toString();
    }
}
