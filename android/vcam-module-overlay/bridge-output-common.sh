#!/system/bin/sh

# Pure helpers shared by the Android output selector and host-side tests.
# Keep this file free of side effects: it is sourced by the long-lived selector.

normalize_output_bool() {
    case "$1" in
        0|1) printf '%s\n' "$1" ;;
        *) return 1 ;;
    esac
}

normalize_output_rotation() {
    case "$1" in
        0|90|180|270) printf '%s\n' "$1" ;;
        *) return 1 ;;
    esac
}

build_output_filter() {
    output_mirror=$(normalize_output_bool "$1") || return 1
    output_rotation=$(normalize_output_rotation "$2") || return 1
    output_width=$3
    output_height=$4
    output_fps=$5

    case "$output_width:$output_height:$output_fps" in
        *[!0-9:]*|:*|*::*|*:) return 1 ;;
    esac

    output_filter=""
    case "$output_rotation" in
        90) output_filter="transpose=1," ;;
        180) output_filter="hflip,vflip," ;;
        270) output_filter="transpose=2," ;;
    esac

    # Camera2 ID 120 has a fixed landscape geometry. A quarter turn makes the
    # source portrait, so fitting the whole frame here would reduce it to a
    # narrow image surrounded by black. Fill the fixed surface instead: keep
    # the source aspect ratio, scale until both output axes are covered, then
    # crop evenly from the excess axis. Mirroring is applied last, in the
    # viewer's final output coordinates.
    output_filter="${output_filter}scale=${output_width}:${output_height}:force_original_aspect_ratio=increase:flags=fast_bilinear:out_range=full"
    output_filter="${output_filter},crop=${output_width}:${output_height}:(iw-ow)/2:(ih-oh)/2"
    if [ "$output_mirror" = "1" ]; then
        output_filter="${output_filter},hflip"
    fi
    printf '%s,fps=%s\n' "$output_filter" "$output_fps"
}
