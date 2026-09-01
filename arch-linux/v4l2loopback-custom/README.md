# Safe v4l2loopback package

This package keeps the small `VIDIOC_QUERYCAP` driver and bus-name overrides
used by the single Deep Live Cam `Xiaomi Cam` output. It deliberately does
**not** patch the kernel device hierarchy or expose a writable sysfs parent
control.

The retired `sysfs-parent.patch` used `device_move()` to attach an already
registered video class device to a USB interface. Shadow mode unbinds and
rebinds that interface, so this created a false lifetime and power-ordering
dependency and made udev events from an unrelated virtual stream follow live
USB topology changes.

Linux 7.1's driver core takes its own parent references in `device_move()` and
`device_del()`, so the audit did **not** establish a use-after-free from the
patch's extra `put_device()` ordering. The patch was removed because the USB
parent spoof was unnecessary, coupled independently managed devices, added
USB-topology traversal to this small test driver, and expanded the late-boot,
replug, suspend, and shutdown failure surface without improving frame delivery.

Build as a normal user and install the result as root:

```bash
makepkg -Csf
sudo pacman -U ./v4l2loopback-custom-0.15.4-3-any.pkg.tar.zst
```

After installation, `modinfo -p v4l2loopback` should list `driver_name` and
`bus_info`, but never `parent_device`.
