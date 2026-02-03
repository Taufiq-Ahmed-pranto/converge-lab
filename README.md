# Depth Anything 3 - 3D Point Cloud Generation Pipeline

Generate 3D point clouds from RGB images using Depth Anything 3 monocular depth estimation, with optional semantic segmentation for labeled 3D scenes.

## 🚀 Overview

This pipeline provides two main workflows:

### 1. Basic Point Cloud Generation (`generate_pointcloud.py`)
- Monocular depth estimation using Depth Anything 3
- Multi-view 3D reconstruction with camera pose estimation
- Confidence-based point filtering
- Statistical outlier removal
- PLY export for external tools

### 2. Segmented Point Cloud Generation (`generate_segmented_pointcloud.py`)
- Everything from basic pipeline, **plus**:
- Semantic segmentation using Mask2Former (150 ADE20K classes)
- 3D points labeled with object classes (wall, floor, chair, table, etc.)
- Segment-colored visualization
- Exportable labels for downstream tasks

---

## 📋 Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 8GB VRAM | RTX 4070 (12GB) or better |
| CUDA | 12.1+ | 12.4+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | 20GB+ free |

### Software Requirements
- **OS**: Linux (tested on Ubuntu 22.04)
- **Python**: 3.10+
- **Conda/Miniconda**: For environment management

---

## 🛠️ Installation

### Step 1: Verify CUDA Installation

```bash
nvidia-smi
nvcc --version
```

Expected output should show CUDA 12.1+ and a compatible NVIDIA driver.

### Step 2: Create Conda Environment

```bash
# Create environment
conda create -n depth_anything_3 python=3.10 -y

# Activate environment
conda activate depth_anything_3
```

### Step 3: Clone Depth Anything 3 Repository

```bash
# Create project directory
mkdir -p ~/Desktop/project/depth_anything
cd ~/Desktop/project/depth_anything

# Clone Depth Anything 3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
```

### Step 4: Install Core Dependencies

```bash
# Install PyTorch with CUDA support and xformers for speed
pip install xformers torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Depth Anything 3 requirements
pip install -r requirements.txt

# Install Depth Anything 3 package
pip install -e .

# Install point cloud and computer vision dependencies
pip install open3d matplotlib scikit-learn opencv-python

# Install segmentation dependencies (for Mask2Former)
pip install transformers accelerate
```

### Step 5: Verify Installation

```bash
python -c "
import torch
from depth_anything_3.api import DepthAnything3
import open3d as o3d
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ Open3D:', o3d.__version__)

# Test model instantiation (lightweight check)
try:
    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-tiny-ade-semantic')
    print('✅ Mask2Former (Transformers): OK')
except Exception as e:
    print('❌ Mask2Former Check Failed:', e)

print('✅ Depth Anything 3: OK')
"
```

---

## 📁 Project Structure

```
depth_anything/
├── generate_pointcloud.py           # Basic point cloud pipeline
├── generate_segmented_pointcloud.py # Segmented point cloud pipeline
├── DATA/                            # Input images
│   ├── SAMPLE_SCENE/               
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── lab_four_picture/           
│   └── your_scene/                  # Your custom scene
├── RESULTS/                         # Output results
│   └── lab_four_picture/
│       ├── scene_pointcloud.ply           # Basic pipeline output
│       ├── scene_pointcloud_rgb.ply       # Segmented: RGB colors
│       ├── scene_pointcloud_segmented.ply # Segmented: class colors
│       └── scene_pointcloud_labels.npz    # Segmented: labels data
└── Depth-Anything-3/                # DA3 repository
```

---

## 🎯 Usage

### Option 1: Basic Point Cloud (RGB only)

```bash
# With visualization
python generate_pointcloud.py --data_folder your_folder_name

# Without visualization
python generate_pointcloud.py --data_folder your_folder_name --no_visualize

# Custom confidence threshold
python generate_pointcloud.py --data_folder your_folder_name --conf_thresh 0.5
```

**Output:** `RESULTS/your_folder_name/scene_pointcloud.ply`

### Option 2: Segmented Point Cloud (RGB + Labels)

```bash
# Basic run
python generate_segmented_pointcloud.py --data_folder your_folder_name

# RECOMMENDED: High-quality alignment (Global registration)
python generate_segmented_pointcloud.py --data_folder your_folder_name --multiway

# MAXIMUM QUALITY: Multiway alignment + Noise filtering + Downsampling
python generate_segmented_pointcloud.py --data_folder your_folder_name --multiway --multiview --voxel_size 0.02
```

**Advanced Flags:**
| Flag | Description | Recommendation |
|------|-------------|----------------|
| `--multiway` | Uses Pose Graph optimization for global alignment | **Highly Recommended** for multi-view |
| `--icp` | Uses pairwise alignment refinement | Good for simple sequences |
| `--multiview`| Keeps only points seen in multiple views | Removes floating noise artifacts |
| `--voxel_size`| Grid size for downsampling (e.g., 0.02 = 2cm) | Speeds up viewer & cleans labels |
| `--conf_thresh`| Depth confidence threshold (0.0-1.0) | Default 0.4 is usually stable |

**Outputs:**
| File | Description |
|------|-------------|
| `scene_pointcloud_rgb.ply` | Point cloud with original RGB colors |
| `scene_pointcloud_segmented.ply` | Point cloud colored by segment class |
| `scene_pointcloud_labels.npz` | NumPy file with points, colors, and segment IDs |

> 💡 **Note on First Run:** Models (~4GB) will be automatically downloaded from Hugging Face on the first execution. Ensure you have an active internet connection.

---

## 📐 Metric vs. Relative Depth

By default, the pipeline uses **Relative Depth** (accurate geometry but arbitrary scale). For real-world measurements in meters:
1. Ensure your dataset has sufficient movement.
2. Edit `generate_segmented_pointcloud.py` to use `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
3. The results in the `.npz` file and `.ply` will then be at metric scale.

---

## 📸 Input Requirements

### Image Guidelines
| Requirement | Recommendation |
|-------------|----------------|
| Format | JPG, JPEG, or PNG |
| Resolution | Any (auto-resized by model) |
| Quantity | 3-20 images for good coverage |
| Overlap | 30-50% between consecutive views |
| Movement | Smooth camera motion around scene |

### Tips for Best Results
1. **Lighting**: Consistent, well-lit conditions
2. **Sharpness**: Avoid motion blur
3. **Coverage**: Capture from multiple angles
4. **Static scene**: Objects should not move between shots

---

## 🔬 How It Works

### Basic Pipeline Flow

```
RGB Images → Depth Anything 3 → Depth Maps + Camera Poses
                                        ↓
                              Back-projection (2D → 3D)
                                        ↓
                              Merge Point Clouds
                                        ↓
                              Clean Outliers → PLY Export
```

### Segmented Pipeline Flow

```
                    ┌─── Mask2Former ──→ Segmentation Maps [H,W]
                    │
RGB Images ─────────┤
                    │
                    └─── Depth Anything 3 ──→ Depth Maps + Poses
                                                    ↓
                              Back-projection (2D → 3D)
                              Each pixel → 3D point with:
                                - XYZ coordinates
                                - RGB color
                                - Segment label
                                        ↓
                              Merge + Clean → Labeled PLY Export
```

### The Key Insight

Every pixel in a 2D image becomes a 3D point. When we back-project:
- **Depth** tells us how far the pixel is from the camera
- **Camera intrinsics** tell us the projection geometry
- **Camera pose** tells us where the camera is in 3D space

So if we know the segment label of a 2D pixel, we simply "carry" that label when converting to 3D:

```
2D Pixel (u, v) → RGB color + Segment ID
        +
    Depth Z
        ↓
3D Point (X, Y, Z, R, G, B, Label)
```

---

## 🏷️ Segmentation Classes (ADE20K)

The segmented pipeline uses Mask2Former trained on ADE20K with **150 classes**:

| ID | Class | ID | Class | ID | Class |
|----|-------|----|---------|----|-------|
| 0 | wall | 5 | ceiling | 15 | table |
| 3 | floor | 10 | cabinet | 19 | chair |
| 14 | door | 24 | shelf | 33 | desk |
| 82 | light | 74 | computer | 65 | toilet |

**Example output from lab scene:**
```
Class ID   Class Name       Points      Percentage
-------------------------------------------------
3          floor            239,658     32.39%
0          wall             200,826     27.14%
5          ceiling          173,617     23.46%
24         shelf            30,659      4.14%
82         light            16,426      2.22%
15         table            15,249      2.06%
19         chair            6,823       0.92%
```

---

## 📊 Loading Segmented Results in Python

```python
import numpy as np
import open3d as o3d

# Load the labels file
data = np.load("RESULTS/your_scene/scene_pointcloud_labels.npz", allow_pickle=True)

points = data['points']           # (N, 3) - XYZ coordinates
colors = data['colors']           # (N, 3) - RGB values [0-1]
segments = data['segments']       # (N,) - Class IDs (0-149)
class_names = data['class_names'] # List of 150 class names

# Example: Extract only "chair" points (class 19)
chair_mask = segments == 19
chair_points = points[chair_mask]
chair_colors = colors[chair_mask]
print(f"Found {len(chair_points)} points belonging to chairs")

# Example: Get all unique classes in the scene
unique_classes = np.unique(segments)
for class_id in unique_classes:
    count = np.sum(segments == class_id)
    print(f"{class_names[class_id]}: {count} points")

# Visualize specific class
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(chair_points)
pcd.colors = o3d.utility.Vector3dVector(chair_colors)
o3d.visualization.draw_geometries([pcd])
```

---

## 🖥️ Visualization Controls

When the Open3D viewer opens:

| Control | Action |
|---------|--------|
| Left Mouse + Drag | Rotate view |
| Right Mouse + Drag | Pan view |
| Scroll Wheel | Zoom in/out |
| `R` | Reset view |
| `Q` or `Esc` | Close viewer |

---

## 🔧 Available Models

### Depth Models (Depth Anything 3)

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| `DA3-SMALL` | ~2GB | Good | Fast |
| `DA3-BASE` | ~4GB | Better | Medium |
| `DA3-LARGE` | ~8GB | Excellent | Slow |
| `DA3-GIANT-1.1` | ~12GB | Best | Slowest |

To change the depth model, edit in the script:
```python
def load_da3_model(model_name="depth-anything/DA3-SMALL"):  # Change here
```

### Segmentation Models (Mask2Former)

| Model | VRAM | Classes |
|-------|------|---------|
| `facebook/mask2former-swin-tiny-ade-semantic` | ~2GB | 150 (ADE20K) |
| `facebook/mask2former-swin-base-ade-semantic` | ~4GB | 150 (ADE20K) |
| `facebook/mask2former-swin-large-ade-semantic` | ~8GB | 150 (ADE20K) |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**For basic pipeline:**
```python
# Edit generate_pointcloud.py - use smaller model
def load_da3_model(model_name="depth-anything/DA3-SMALL"):
```

**For segmented pipeline:**
- The script automatically manages memory by loading models sequentially
- Reduce image resolution or number of images if still failing

### No Images Found
```bash
# Ensure images are in correct location:
ls DATA/your_folder_name/
# Should show: image1.jpg, image2.jpg, etc.
```

### Visualization Not Working
```bash
# For headless servers, disable visualization:
python generate_pointcloud.py --data_folder my_scene --no_visualize
python generate_segmented_pointcloud.py --data_folder my_scene --no_visualize
```

### Poor Point Cloud Quality
- Increase confidence threshold: `--conf_thresh 0.5` or `0.6`
- Use more overlapping images (minimum 4 recommended)
- Ensure good, consistent lighting
- Avoid motion blur in source images

### Segmentation Not Accurate
- Mask2Former works best with clear object boundaries
- Very cluttered scenes may have lower accuracy
- Consider using SAM for interactive refinement

---

## 📤 Viewing Output Files

### PLY Files
- **MeshLab** (free): `meshlab scene_pointcloud.ply`
- **CloudCompare** (free): File → Open
- **Blender** (free): File → Import → PLY
- **Open3D Python**:
  ```python
  import open3d as o3d
  pcd = o3d.io.read_point_cloud("scene_pointcloud.ply")
  o3d.visualization.draw_geometries([pcd])
  ```

### NPZ Labels File
```python
import numpy as np
data = np.load("scene_pointcloud_labels.npz", allow_pickle=True)
print(data.files)  # ['points', 'colors', 'segments', 'class_names']
```

---

## 📚 API Reference

### Basic Pipeline Functions

```python
from generate_pointcloud import (
    setup_paths,                  # Create data/results paths
    load_da3_model,               # Load Depth Anything 3 model
    load_images_from_folder,      # Find images in folder
    run_da3_inference,            # Run depth estimation
    merge_point_clouds,           # Combine depth maps to 3D
    clean_point_cloud_open3d,     # Remove outliers
    visualize_point_cloud_open3d, # 3D visualization
    export_point_cloud_ply,       # Save to PLY file
    process_pipeline,             # Run complete pipeline
)
```

### Segmented Pipeline Functions

```python
from generate_segmented_pointcloud import (
    load_segmentation_model,              # Load Mask2Former
    run_segmentation,                     # Run segmentation on image
    merge_point_clouds_with_segments,     # Combine with labels
    clean_point_cloud_with_segments,      # Clean while preserving labels
    export_point_cloud_with_labels,       # Export RGB + segmented + NPZ
    visualize_segmented_point_cloud,      # View with segment colors
    print_segment_statistics,             # Print class distribution
    process_pipeline_with_segmentation,   # Run complete pipeline
)
```

---

## 🎓 Example Workflow

```bash
# 1. Activate environment
conda activate depth_anything_3

# 2. Place your images
mkdir -p DATA/my_room
cp /path/to/photos/*.jpg DATA/my_room/

# 3. Generate basic point cloud
python generate_pointcloud.py --data_folder my_room

# 4. Generate segmented point cloud
python generate_segmented_pointcloud.py --data_folder my_room

# 5. View results
ls RESULTS/my_room/
# scene_pointcloud.ply
# scene_pointcloud_rgb.ply
# scene_pointcloud_segmented.ply
# scene_pointcloud_labels.npz
```

---

## 📄 License

This project uses:
- **Depth Anything 3** - Apache License 2.0
- **Mask2Former** - MIT License
- **Open3D** - MIT License

---

## 🙏 Acknowledgments

- **ByteDance** for Depth Anything 3
- **Meta AI (FAIR)** for Mask2Former
- **Open3D Team** for 3D visualization
- **Hugging Face** for Transformers library

---

**Last Updated**: January 17, 2026  
**Tested Environment**: Ubuntu 22.04, CUDA 12.4, Python 3.10, PyTorch 2.x, RTX 4070