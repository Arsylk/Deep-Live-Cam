package dev.vcam.bridge;

import android.Manifest;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class MainActivity extends Activity {
    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private TextView statusView;

    private final BroadcastReceiver statusReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String status = intent.getStringExtra(CameraBridgeService.EXTRA_STATUS);
            if (status != null) {
                statusView.setText(status);
            }
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(40, 72, 40, 40);

        TextView title = new TextView(this);
        title.setText("Deep Live Camera Bridge");
        title.setTextSize(24.0f);
        layout.addView(title);

        statusView = new TextView(this);
        statusView.setText("Camera2 → GPU orientation → hardware H.264 → Windows 192.168.1.35\n"
                + "Processed return appears as camera 120. Orientation defaults to Auto and can "
                + "be changed in the Arch control center.");
        statusView.setTextSize(16.0f);
        statusView.setPadding(0, 32, 0, 32);
        layout.addView(statusView);

        Button start = new Button(this);
        start.setText("Start bridge");
        start.setOnClickListener(view -> ensurePermissionAndStart());
        layout.addView(start);

        Button stop = new Button(this);
        stop.setText("Stop bridge");
        stop.setOnClickListener(view -> {
            Intent intent = new Intent(this, CameraBridgeService.class);
            intent.setAction(CameraBridgeService.ACTION_STOP);
            startService(intent);
            statusView.setText("Bridge stopped; camera 120 remains on its fallback frame.");
        });
        layout.addView(stop);

        setContentView(layout);
        ensurePermissionAndStart();
    }

    private void ensurePermissionAndStart() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 101);
        }
        Intent intent = new Intent(this, CameraBridgeService.class);
        intent.setAction(CameraBridgeService.ACTION_START);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        statusView.setText("Starting physical-camera encoder…");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                ensurePermissionAndStart();
            } else {
                statusView.setText("Camera permission is required for the bridge input.");
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        IntentFilter filter = new IntentFilter(CameraBridgeService.ACTION_STATUS);
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(statusReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(statusReceiver, filter);
        }
    }

    @Override
    protected void onStop() {
        unregisterReceiver(statusReceiver);
        super.onStop();
    }
}
