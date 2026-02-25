# HECD 3D Computer Vision: Multi-Camera Human Pose Estimation & 3D Reconstruction

A step-by-step tutorial series for extracting 2D human pose from synchronized multi-camera video, matching identities across views, and reconstructing 3D skeletons via triangulation.

## Pipeline Overview

```
  Raw Video (N synchronized cameras)
          │
          ▼
┌─────────────────────────┐
│  Tutorial 1             │
│  YOLOv8 Pose Estimation │   Per-camera 2D keypoint detection
│  (17 COCO joints)       │   Handles landscape & portrait cameras
└────────────┬────────────┘
             │  .slp / .json / .pkl per camera
             ▼
┌─────────────────────────┐
│  Tutorial 2             │
│  Person Re-ID           │   Cross-camera identity matching
│  (OSNet embeddings)     │   "Who in CAM A is who in CAM B?"
└────────────┬────────────┘
             │  identity correspondence map
             ▼
┌─────────────────────────┐
│  Tutorial 3             │
│  3D Triangulation       │   Fuse matched 2D poses into 3D
│  (DLT + calibration)    │   Scene normalization (floor = Z=0)
└────────────┬────────────┘
             │
             ▼
    3D skeleton coordinates
    (n_frames × n_tracks × 17 joints × 3)
```

**Why this order?** Each camera independently detects people but has no idea which detection in Camera A is the same person as in Camera C. Re-identification must come before triangulation — otherwise triangulation doesn't know which 2D keypoints across cameras to combine into 3D points.

---

## Tutorials

### Tutorial 1: Human Pose Estimation with YOLO & Ultralytics

**Goal:** Extract 2D skeleton keypoints from each camera's video independently.

| | |
|---|---|
| **Notebook** | `tutorial_01_yolo_pose_estimation.ipynb` |
| **Model** | YOLOv8n-pose (nano, ~6 MB) |
| **Keypoints** | 17 COCO joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| **Output** | SLEAP `.slp`, JSON, and Pickle per video |

**What you'll learn:**
- Environment setup (conda + CUDA/MPS)
- Extracting frames from video at specific timestamps
- Single-frame inference and understanding YOLO output (boxes, keypoints, confidence)
- Batch processing entire videos with streaming inference
- Handling portrait/rotated cameras (rotate before inference, remap keypoints back)
- Exporting to multiple formats for downstream use

**COCO 17-Keypoint Skeleton:**
```
         nose(0)
        /     \
   l_eye(1)  r_eye(2)
      |         |
   l_ear(3)  r_ear(4)

   l_shoulder(5)───r_shoulder(6)
      |                  |
   l_elbow(7)        r_elbow(8)
      |                  |
   l_wrist(9)        r_wrist(10)

   l_shoulder(5)───r_shoulder(6)
      |                  |
    l_hip(11)──────r_hip(12)
      |                  |
   l_knee(13)        r_knee(14)
      |                  |
   l_ankle(15)       r_ankle(16)
```

---

### Tutorial 2: Person Re-Identification Across Cameras

**Goal:** Determine which detection in Camera A corresponds to which detection in Camera B (and all other cameras).

| | |
|---|---|
| **Notebook** | `tutorial_02_person_reid.ipynb` |
| **Model** | OSNet (Omni-Scale Network, `osnet_x0_25`) |
| **Embeddings** | 512-dimensional appearance vectors per person crop |
| **Matching** | Cosine similarity across camera pairs |
| **Output** | Per-frame cross-camera identity correspondence map |

**What you'll learn:**
- Cropping person images from video using bounding boxes and pose data
- Torso/shirt cropping (shoulders to hips) for robust matching in occluded scenes
- Extracting ReID embeddings with OSNet (~1ms per crop on GPU)
- Computing cosine similarity to measure appearance match quality
- Building per-frame identity maps linking the same person across all cameras

**Why ReID before triangulation?** Without identity matching, 5 people in Camera A and 5 in Camera C produce 120 possible pairings — only 5 are correct. ReID resolves this ambiguity so triangulation gets the right 2D correspondences.

---

### Tutorial 3: Multi-Camera 3D Triangulation

**Goal:** Combine matched 2D poses from all cameras into 3D skeleton coordinates.

| | |
|---|---|
| **Notebook** | `tutorial_03_3d_triangulation.ipynb` |
| **Method** | Direct Linear Transformation (DLT) |
| **Calibration** | Camera intrinsics + extrinsics (TOML format) |
| **Output** | `points3d.h5` — shape `(n_frames, n_tracks, 17, 3)` |

**What you'll learn:**
- Understanding camera calibration files (intrinsic and extrinsic parameters)
- Input data structure: 2D poses (Tutorial 1) + identity map (Tutorial 2)
- Running DLT triangulation with sleap-anipose
- Inspecting 3D results and per-joint reprojection error
- 3D skeleton visualization with matplotlib
- Scene normalization: rotating so Z-axis = up and floor = Z=0

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended)
- Apple Silicon Mac with MPS (supported, slower)
- CPU-only (supported, slowest)

### Software
| Package | Purpose |
|---|---|
| `pytorch` + `torchvision` | Deep learning framework |
| `ultralytics` | YOLOv8 pose estimation |
| `opencv-python` | Video I/O and image manipulation |
| `sleap-io` | SLEAP format reading/writing |
| `boxmot` / `torchreid` | Person re-identification |
| `sleap-anipose` | 3D triangulation engine |
| `numpy`, `scipy`, `matplotlib` | Numerical computation and visualization |
| `h5py` | HDF5 file I/O |

### Quick Start

```bash
# Create environment
conda create -n multicam-pose python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda activate multicam-pose

# Install packages
pip install ultralytics opencv-python sleap-io numpy tqdm jupyter ipykernel
pip install boxmot torchreid scikit-learn
pip install sleap-anipose h5py matplotlib

# Register Jupyter kernel
python -m ipykernel install --user --name multicam-pose --display-name "Python (multicam-pose)"
```

---

## Input Data

These tutorials expect **synchronized multi-camera video** — multiple cameras recording the same scene from different angles, with frames aligned in time. Video files can be `.mp4` or `.mkv`.

For 3D triangulation (Tutorial 3), you also need **camera calibration** files specifying each camera's intrinsic parameters (focal length, distortion) and extrinsic parameters (position and orientation in world coordinates).

## Output Data

| Stage | Format | Description |
|---|---|---|
| Tutorial 1 | `.slp`, `.json`, `.pkl` | 2D keypoints per person per frame per camera |
| Tutorial 2 | `.json`, `.pkl` | Cross-camera identity correspondence per frame |
| Tutorial 3 | `.h5` | 3D coordinates: `(n_frames, n_tracks, 17_joints, 3_xyz)` |

