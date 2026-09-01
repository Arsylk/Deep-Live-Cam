package dev.vcam.mobile;

final class Face {
    final float[] bbox;
    final float[] keypoints;
    final float score;

    Face(float[] bbox, float[] keypoints, float score) {
        this.bbox = bbox;
        this.keypoints = keypoints;
        this.score = score;
    }

    float width() { return Math.max(0.0f, bbox[2] - bbox[0]); }
    float height() { return Math.max(0.0f, bbox[3] - bbox[1]); }
    float area() { return Math.max(0.0f, width() + 1.0f) * Math.max(0.0f, height() + 1.0f); }
}
