# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# Labeled whole-frame quality automation for network live mode.
quality_mode: str = "balanced"  # monitor, balanced, or strict
quality_auto_correct: bool = True
processing_enabled: bool = True
processing_off_output: str = "passthrough"

# Motion-aware live face tracking.  Detection remains authoritative; optical
# flow bridges detector cadence and short misses while these bounds prevent a
# stale face from being held indefinitely.
tracking_enabled: bool = True
detection_interval: int = 1
tracking_smoothing: float = 0.65
tracking_grace_frames: int = 5
minimum_detection_score: float = 0.45
minimum_face_size: int = 64
color_match_strength: float = 0.35

# Frequency-domain repair (anti-detector post-processing)
repair_hf_strength: float = 0.0         # HF noise restoration (0=off, 0.5=max)
repair_checkerboard: float = 0.0        # Checkerboard attenuation (0=off, 1.0=max)
repair_wavelet: float = 0.0             # Wavelet stat matching (0=off, 1.0=max)
repair_boundary_mask: bool = False       # Content-aware boundary mask
repair_boundary_strength: float = 0.0    # Target-preserving seam reblend (0-1)
# Deprecated compatibility field. Direct target-texture transfer failed the
# measured quality gates and is intentionally not applied or shown in the UI.
repair_skin_texture: float = 0.0
repair_camera_detail: float = 0.0        # Final-resolution adaptive detail match (0-4)

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
swapper_model: str = "auto"         # auto, inswapper-128, instyle-256, simswap-512, or native-256
swapper_backend: str = "auto"       # auto, ort, or ncnn (independent of ORT EP)
active_swapper_model: str = "not-loaded"
active_swapper_backend: str = "not-loaded"
active_swapper_resolution: int = 0   # native generated face edge, not output canvas
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
