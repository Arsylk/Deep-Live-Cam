<h1 align="center">Deep-Live-Cam 2.1.6</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

This fork extends the upstream real-time face-swap project into a complete
two-host system: an Arch Linux workstation (slot 1) paired with an Android
phone (Apollo, Mi 10T) over SRT, plus a phone-side global front-camera
substitution module. The upstream GUI/face-swap core is untouched; everything
below is additive or a contained modification.

## New distinct components

### 1. Android capture app — `android/vcam-app/` (APK, v1.7)
- Camera2 capture → OpenGL stage → hardware H.264 (c2.qti.avc) → local TCP →
  on-device ffmpeg → SRT to Arch. Streams the front camera's **native
  portrait transport 720x1280@30** (720p-class, ~10 Mbps, B-frame-free).
- The GL stage renders aspect-true: the sensor's full landscape FOV is
  rotated upright into the portrait canvas — no crop, no zoom, no bars.
- Allow-listed live controls (set from the Arch manager or the app UI):
  lens selection, rotation, zoom, exposure compensation, AE/AWB locks,
  stabilization.
- Runs a processed-return receiver and publishes the return as processed
  Camera2 ID 120 on the phone (with `vcam-module-overlay`).

### 2. Xposed module — `android/vcam-camera-logger/` (APK, v0.4.2)
Global system front camera for the phone:
- Discovers **every** front-facing Camera2 device at runtime and replaces it
  with the processed stream (ID 120) — no per-app tables; works on
  multi-front devices.
- Camera metadata spoofing for app compatibility: LENS_FACING=FRONT, the real
  front camera's AE FPS ranges, LIMITED hardware level, and CamcorderProfile
  redirection — CameraX/Aperture video-quality validation passes cleanly.
- **Replace, never add**: the processed camera occupies the front's slot in
  the enumeration (one fewer entry), verified by host tests.
- Routes the returned webcam audio as the session microphone (Remote Submix,
  native CAPTURE_AUDIO_OUTPUT grant for all ordinary app UIDs).
- Global kill-switch file `/data/local/tmp/vcam_disable` — instant native
  fallback without uninstalling. Host-testable pure routing policy
  (`CameraRoutingPolicy`, 15+ tests).

### 3. Magisk module — `android/vcam-module-overlay/`
v4l2loopback `/dev/video20` + AOSP external-camera provider config exposing
the processed return as Camera2 ID 120; VINTF vendor binds only.

### 4. Arch native manager GUI — `arch-linux/bin/dlc_manager/`
PySide6 desktop app (runs as root via pkexec desktop entries):
- **Live** workspace: camera previews, wheel-guarded controls.
- **Input** tab: semantic source selection (Arch webcam / phone front /
  phone back / prerecorded / **assembler**) with per-input device cards.
- **Render** workspace: record → offline high-quality render (graded) →
  replay as a camera input; file management.
- **Assembler input card**: compose prompt sequences from the puppet
  library, assemble & load in <1 s; full prerecorded-style preview
  (aspect-true, drag/zoom, grid, play/pause/seek).
- Phone-route reconciliation: single-owner serialization of the phone's one
  SRT return endpoint across local/Windows/direct modes.

### 5. Puppet prompt system — `arch-linux/bin/puppet_*.py`
- `puppet_recorder.py`: guided recording GUI (16-action session, cue sheet,
  live activity log, live preview during takes via an encoder tee).
- `puppet_library_build.py`: recording → offline face-swap (A-/B+ graded
  pipeline) → cue-based segment cutting → concat-safe segment library
  (verified: keyframes every 12, no B-frames).
- `puppet_assemble.py`: prompt → segment selection → ffmpeg concat-copy.
  Any prompt sequence ("turn_left, blink, say 4-7-2") becomes a playing
  camera source in about a second, no inference at prompt time.
- `/opt/github/LivePortrait` (sibling checkout, not in this repo): parameter-
  driven fallback renderer for novel motions.

### 6. Offline render + quality grading
`arch-linux/bin/offline_renderer.py` + `render_quality_score.py`: full-
pipeline offline face swap with AI auto-tuning, GFPGAN enhancement option,
swap-relative realism scoring (seam/detail deltas) and ArcFace identity
scoring, banded grades (A+ … F).

## Major modifications to existing pieces

- **Native portrait phone transport**: the phone→Arch stream is the camera's
  native 720x1280@30, carried unmodified to the processor and preview taps;
  the only reshape is the fit into the locked 1280x720 system-camera output
  at delivery. Processor width/height fully parameterized.
- **Receiver**: file_relay input (reads MP4s directly), gapless prerecorded
  looping via `-stream_loop` (no black flash), once/freeze playback modes,
  live pan/zoom over zmq (crop@live, no decoder restart), seek + pause
  control files, aspect-true fitting of any input at the locked output.
- **Sender**: VAAPI encode with periodic forced keyframes (fixes mid-stream
  join corruption), three local taps (11000/11001/11002) + processor source
  tap (11005).
- **Camera adapters**: allow-listed controls persisted to the phone app and
  applied on input selection (lens, rotation, zoom, exposure, stabilization).
- **Verified port map**: all 14 UDP taps + 5 SRT endpoints documented with
  producer/consumer ownership in `arch-linux/README.md` ("Fixed routes");
  single-reader rule enforced by convention and tested.
- **Tests**: 300+ host tests covering the manager, routing policy, receiver
  routes, prerecorded/assembler behavior, and the phone pipeline.

## Deployed layout (production phone + Arch)

- Arch: services (`deep-live-cam-{sender,receiver,phone-processed,
  phone-return-relay}.service`), installed manager + helpers under
  `/usr/local/lib/deep-live-cam-arch/`, desktop entries with pkexec.
- Phone: vcam-app v1.7 + Xposed module v0.4.2 (LSPosed/Vector) + overlay
  module; global front camera with kill switch.

See `arch-linux/README.md` for the full pipeline topology and the verified
port map, `arch-linux/PRERECORDED_WORKFLOW.md` for the record/render/replay
workflow, and `.remember/` (local, untracked) for session history.

---



<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## Pre-built Quickstart

<p align="center">
  <a href="https://deeplivecam.net/index.php/quickstart">
    <img src="https://github.com/user-attachments/assets/fa2cdf79-c933-4b93-844a-b087192261ed" width="100%" alt="Lite / Ultimate Download Banner">
  </a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/56b61811-3a1e-4672-9b50-cf7f6e8e6852" width="40" alt="Windows">
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/6538e3a6-c957-431a-b586-2d6abcf534dc" width="34" alt="Mac Silicon">
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/ad45142e-426c-4364-a2a9-a512670cc62c" width="40" alt="CPU">
</p>

<p align="center">
  <strong>Windows • Mac Silicon • CPU • NVIDIA • AMD</strong>
</p>

<p align="center">
  Builds optimized for your hardware.
</p>

<p align="center">
  <a href="https://deeplivecam.net/index.php/quickstart">
    <img src="media/Download.png" width="280" alt="Download">
  </a>
</p>

> **Ultimate** includes **30+ exclusive features**, performance optimizations, and **priority support**.

Perfect if you want the fastest setup with **zero manual installation**, pre-configured dependencies, and optimized builds for every supported platform.

## TLDR; Live Deepfake in just 3 Clicks
![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)
1. Select a face
2. Select which camera to use
3. Press live!

## Features & Uses - Everything is in real-time

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450"> 
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

### Omegle

**Surprise people on Omegle**

<p align="center">
  <video src="https://github.com/user-attachments/assets/2e9b9b82-fa04-4b70-9f56-b1f68e7672d0" width="450" controls></video>
</p>

## Installation (Manual)

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the quickstart version.**

<details>
<summary>Click to see the process</summary>

### Installation

This is more likely to work on your computer but will be slower as it utilizes the CPU.

**1. Set up Your Platform**

-   Python (3.14 recommended; 3.11-3.14 supported)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) - ```iex (irm ffmpeg.tc.ht)```
-   [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**2. Clone the Repository**

```bash
git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**3. Download the Models**

1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx)
2. [inswapper\_128\_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies**

We highly recommend using a `venv` to avoid issues.


For Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
For Linux:
```bash
# Ensure you use the installed Python 3.14
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For macOS:**

Apple Silicon (M1 through M5) requires specific setup:

```bash
# Install Python 3.14
brew install python@3.14

# Install tkinter package (required for the GUI)
brew install python-tk@3.14

# Create and activate virtual environment with Python 3.14
python3.14 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

** In case something goes wrong and you need to reinstall the virtual environment **

```bash
# Deactivate the virtual environment
rm -rf venv

# Reinstall the virtual environment
python -m venv venv
source venv/bin/activate

# install the dependencies again
pip install -r requirements.txt

# gfpgan and basicsrs issue fix
pip install git+https://github.com/xinntao/BasicSR.git@master
pip uninstall gfpgan -y
pip install git+https://github.com/TencentARC/GFPGAN.git@master
```

**Run:** If you don't have a GPU, you can run Deep-Live-Cam using `python run.py`. Note that initial execution will download models (~300MB).

### GPU Acceleration

**CUDA Execution Provider (Nvidia)**

1. Install [CUDA Toolkit 12.8.0](https://developer.nvidia.com/cuda-12-8-0-download-archive)
2. Install [cuDNN v8.9.7 for CUDA 12.x](https://developer.nvidia.com/rdp/cudnn-archive) (required for onnxruntime-gpu):
   - Download cuDNN v8.9.7 for CUDA 12.x
   - Make sure the cuDNN bin directory is in your system PATH
3. Install dependencies:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.21.0
```

3. Usage:

```bash
python run.py --execution-provider cuda
```

**CoreML Execution Provider (Apple Silicon)**

Apple Silicon (M1 through M5) specific installation:

1. Make sure you've completed the macOS setup above using Python 3.14.
2. No extra install step is needed — `requirements.txt` pulls the official
   `onnxruntime` build, whose macOS wheels ship the CoreML execution provider.
   If you previously installed the unmaintained `onnxruntime-silicon` fork,
   remove it first, as it shadows the real package:

```bash
pip uninstall onnxruntime-silicon
pip install -r requirements.txt
```

3. Usage:

```bash
python3.14 run.py --execution-provider coreml
```

**Important Notes for macOS:**
- Python 3.11 is the minimum (onnxruntime dropped 3.10); 3.14 is recommended
- Always run with `python3.14` command not just `python` if you have multiple Python versions installed
- If you get error about `_tkinter` missing, reinstall the tkinter package: `brew reinstall python-tk@3.14`
- If you get model loading errors, check that your models are in the correct folder
- If you encounter conflicts with other Python versions, consider uninstalling them:
  ```bash
  # List all installed Python versions
  brew list | grep python

  # Uninstall conflicting versions if needed
  brew uninstall --ignore-dependencies python@3.11

  # Keep only Python 3.14
  brew cleanup
  ```

**CoreML Execution Provider (Apple Legacy)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.21.0
```

2. Usage:

```bash
python run.py --execution-provider coreml
```

**DirectML Execution Provider (Windows)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.21.0
```

2. Usage:

```bash
python run.py --execution-provider directml
```

**OpenVINO™ Execution Provider (Intel)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.21.0
```

2. Usage:

```bash
python run.py --execution-provider openvino
```

**ncnn Vulkan Swapper (Linux/AMD, including Polaris)**

This independent backend uses Mesa Vulkan for the expensive face-swap model
while keeping detection and recognition on the selected ONNX Runtime provider.
After the one-time local preparation in
[arch-linux/ncnn/README.md](arch-linux/ncnn/README.md), it is selected
automatically and requires no network access at runtime.

```bash
python run.py --execution-provider cpu --swapper-backend ncnn
```

**Distilled native-256 swapper (development)**

The repository now includes the complete `DLC-Swap256-M` architecture,
authorized-data loader, distillation losses, ONNX exporter, hash-pinned local
runtime contract, learned semantic fusion mask, and application integration.
It does not include an unqualified or unlicensed trained checkpoint. See
[native256/README.md](native256/README.md) for training and release gates, and
[MOBILEFACESWAP_BASELINE.md](MOBILEFACESWAP_BASELINE.md) for the separately
licensed experimental teacher/baseline path.

An explicitly installed development bundle can be exercised with:

```bash
python run.py --swapper-model native-256 --swapper-backend ort
# Linux/Vulkan after the offline package step in native256/README.md:
python run.py --swapper-model native-256 --swapper-backend ncnn
```

`--swapper-model auto` admits native-256 only when its local manifest is marked
`qualified`, explicitly auto-eligible, and backed by a hash-pinned qualification
report covering every deployed artifact and release gate; otherwise it keeps
the current INSwapper path. Runtime model selection, validation, and inference
perform no downloads. Explicit `ncnn` never silently substitutes ORT or a
different model; backend `auto` may fall back to the same native-256 ONNX graph.
</details>

## Usage

**1. Image/Video Mode**

-   Execute `python run.py`.
-   Choose a source face image and a target image/video.
-   Click "Start".
-   The output will be saved in a directory named after the target video.

**2. Webcam Mode**

-   Execute `python run.py`.
-   Select a source face image.
-   Click "Live".
-   Wait for the preview to appear (10-30 seconds).
-   Use a screen capture tool like OBS to stream.
-   To change the face, select a new source image.

## Download all models in this huggingface link
- [**Download models here**](https://huggingface.co/hacksider/deep-live-cam/tree/main)

## Command Line Arguments (Unmaintained)

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            the live camera display as you see it in the front-facing camera frame
  --live-resizable                                         the live camera frame is resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  --swapper-model {auto,inswapper-128,instyle-256,simswap-512,native-256} local face-swap model family
  --swapper-backend {auto,ort,ncnn}                        face-swap inference backend
  -v, --version                                            show program's version number and exit
```

Looking for a CLI mode? Using the -s/--source argument will make the run program in cli mode.

## Press

 - [**Ars Technica**](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - *"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"*
 - [**Yahoo!**](https://www.yahoo.com/tech/ok-viral-ai-live-stream-080041056.html) - *"OK, this viral AI live stream software is truly terrifying"*
 - [**CNN Brasil**](https://www.cnnbrasil.com.br/tecnologia/ia-consegue-clonar-rostos-na-webcam-entenda-funcionamento/) - *"AI can clone faces on webcam; understand how it works"*
 - [**Bloomberg Technoz**](https://www.bloombergtechnoz.com/detail-news/71032/kenalan-dengan-teknologi-deep-live-cam-bisa-jadi-alat-menipu) - *"Get to know Deep Live Cam technology, it can be used as a tool for deception."*
 - [**TrendMicro**](https://www.trendmicro.com/vinfo/gb/security/news/cyber-attacks/ai-vs-ai-deepfakes-and-ekyc) - *"AI vs AI: DeepFakes and eKYC"*
 - [**PetaPixel**](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - *"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*
 - [**SomeOrdinaryGamers**](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - *"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*
 - [**IShowSpeed**](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - *"Alright look look look, now look chat, we can do any face we want to look like chat"*
 - [**TechLinked (Linus Tech Tips)**](https://www.youtube.com/watch?v=wnCghLjqv3s&t=551s) - *"They do a pretty good job matching poses, expression and even the lighting"*
 - [**IShowSpeed**](https://youtu.be/JbUPRmXRUtE?t=3964) - *"What the F***! Why do I look like Vinny Jr? I look exactly like Vinny Jr!? No, this shit is crazy! Bro This is F*** Crazy!"*


## Credits

-   [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
-   [Henry](https://github.com/henryruhs): One of the major contributor in this repo
-   [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
-   [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
-   [GosuDRM](https://github.com/GosuDRM): for the open version of roop
-   [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
-   [vic4key](https://github.com/vic4key): For supporting/contributing to this project
-   [kier007](https://github.com/kier007): for improving the user experience
-   [qitianai](https://github.com/qitianai): for multi-lingual support
-   [laurigates](https://github.com/laurigates): Decoupling stuffs to make everything faster!
-   [maxwbuckley](https://github.com/maxwbuckley): For making the effort to optimize this for mac!
-   and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
-   Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
-   All the wonderful users who helped make this project go viral by starring the repo ❤️

[![Stargazers](https://reporoster.com/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

## Stars to the Moon 🚀

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
