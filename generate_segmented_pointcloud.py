"""
Segmented 3D Point Cloud Generation Pipeline
Combines Depth Anything 3 depth estimation with semantic segmentation
to create labeled 3D point clouds.
"""

import glob
import os
import torch
import numpy as np
import open3d as o3d
import argparse
from PIL import Image
from depth_anything_3.api import DepthAnything3
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ADE20K class labels (150 classes) - common indoor/outdoor objects
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]


def get_segment_colors(num_segments):
    """Generate distinct colors for each segment"""
    np.random.seed(42)  # Reproducible colors
    colors = np.random.rand(num_segments, 3)
    # Make first few colors more distinct
    distinct_colors = np.array([
        [0.9, 0.1, 0.1],  # Red
        [0.1, 0.9, 0.1],  # Green
        [0.1, 0.1, 0.9],  # Blue
        [0.9, 0.9, 0.1],  # Yellow
        [0.9, 0.1, 0.9],  # Magenta
        [0.1, 0.9, 0.9],  # Cyan
        [0.9, 0.5, 0.1],  # Orange
        [0.5, 0.1, 0.9],  # Purple
        [0.1, 0.5, 0.1],  # Dark Green
        [0.5, 0.5, 0.5],  # Gray
    ])
    colors[:min(len(distinct_colors), num_segments)] = distinct_colors[:min(len(distinct_colors), num_segments)]
    return colors


def setup_paths(data_folder="SAMPLE_SCENE", base_path=None):
    """Create project paths for data, results, and models"""
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    paths = {
        'data': os.path.join(base_path, 'DATA', data_folder),
        'results': os.path.join(base_path, 'RESULTS', data_folder),
    }
    os.makedirs(paths['results'], exist_ok=True)
    return paths


def load_da3_model(model_name="depth-anything/DA3-GIANT-1.1"):
    """Initialize Depth-Anything-3 model on available device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    return model, device


def load_segmentation_model(model_name="facebook/mask2former-swin-base-ade-semantic"):
    """Initialize Mask2Former segmentation model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading segmentation model: {model_name}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def load_images_from_folder(data_path, extensions=['*.jpg', '*.png', '*.jpeg']):
    """Scan folder and load all images with supported extensions"""
    image_files = []
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist!")
        return []
        
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(data_path, ext))))
    
    print(f"Found {len(image_files)} images in {data_path}")
    return image_files


def run_segmentation(seg_model, processor, image_path, device):
    """Run semantic segmentation on a single image"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Resize large images to avoid OOM
    max_size = 512
    if max(original_size) > max_size:
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image_resized = image.resize(new_size, Image.BILINEAR)
        target_size = (new_size[1], new_size[0])  # (height, width)
    else:
        image_resized = image
        target_size = (original_size[1], original_size[0])
    
    inputs = processor(images=image_resized, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    # Get semantic segmentation map at resized resolution (not original to save memory)
    predicted_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[target_size]
    )[0]
    
    seg_map = predicted_map.cpu().numpy()
    
    # Resize back to original size using PIL (CPU, memory efficient)
    if max(original_size) > max_size:
        seg_pil = Image.fromarray(seg_map.astype(np.int32))
        seg_pil = seg_pil.resize(original_size, Image.NEAREST)
        seg_map = np.array(seg_pil)
    
    return seg_map


def run_da3_inference(model, image_files, process_res_method="upper_bound_resize"):
    """Run Depth-Anything-3 to get depth maps, camera poses, and intrinsics"""
    prediction = model.inference(
        image=image_files,
        infer_gs=True,
        process_res_method=process_res_method
    )
    
    print(f"Depth maps shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")
    
    return prediction


def resize_segmentation_to_depth(seg_map, target_shape):
    """Resize segmentation map to match depth map resolution using nearest neighbor"""
    from PIL import Image
    seg_pil = Image.fromarray(seg_map.astype(np.int32))
    seg_resized = seg_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return np.array(seg_resized)


def depth_to_point_cloud_with_segments(depth_map, rgb_image, intrinsics, extrinsics, 
                                        seg_map, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points with segmentation labels"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter by confidence if provided
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u = u[valid_mask]
        v = v[valid_mask]
        depth_flat = depth_map[valid_mask]
        rgb_flat = rgb_image[valid_mask]
        seg_flat = seg_map[valid_mask]
    else:
        u = u.flatten()
        v = v.flatten()
        depth_flat = depth_map.flatten()
        rgb_flat = rgb_image.reshape(-1, 3)
        seg_flat = seg_map.flatten()
    
    # Back-project to camera coordinates
    x = (u - cx) * depth_flat / fx
    y = (v - cy) * depth_flat / fy
    z = depth_flat
    
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates using extrinsics (w2c format)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R  # Inverse transform
    
    colors = rgb_flat.astype(np.float32) / 255.0
    
    return points_world, colors, seg_flat


def merge_point_clouds_with_segments(prediction, seg_maps, conf_thresh=0.5):
    """Combine all frames into single point cloud with segmentation labels"""
    all_points = []
    all_colors = []
    all_segments = []
    
    n_frames = len(prediction.depth)
    
    for i in range(n_frames):
        # Resize segmentation to match depth map
        depth_shape = prediction.depth[i].shape
        seg_resized = resize_segmentation_to_depth(seg_maps[i], depth_shape)
        
        points, colors, segments = depth_to_point_cloud_with_segments(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            seg_resized,
            prediction.conf[i],
            conf_thresh
        )
        all_points.append(points)
        all_colors.append(colors)
        all_segments.append(segments)
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    merged_segments = np.concatenate(all_segments)
    
    print(f"Merged point cloud: {len(merged_points)} points")
    print(f"Unique segments found: {len(np.unique(merged_segments))}")
    
    return merged_points, merged_colors, merged_segments


def clean_point_cloud_with_segments(points_3d, colors_3d, segments, nb_neighbors=20, std_ratio=2.0):
    """Cleans a point cloud while preserving segment labels"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if colors_3d.max() > 1.0:
        pcd.colors = o3d.utility.Vector3dVector(colors_3d / 255.0)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors_3d)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_mask = np.asarray(ind)
    
    cleaned_points = points_3d[inlier_mask]
    cleaned_colors = colors_3d[inlier_mask]
    cleaned_segments = segments[inlier_mask]

    return cleaned_points, cleaned_colors, cleaned_segments


def export_segmented_point_cloud_ply(points, colors, segments, filepath):
    """Export point cloud with segment colors to PLY format"""
    # Create segment-colored version
    unique_segments = np.unique(segments)
    segment_colors = get_segment_colors(int(segments.max()) + 1)
    
    colored_by_segment = segment_colors[segments.astype(int)]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colored_by_segment)
    
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Segmented point cloud exported to {filepath}")


def export_point_cloud_with_labels(points, colors, segments, filepath_base):
    """Export point cloud with original colors and segment labels as separate files"""
    # Original colors PLY
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"{filepath_base}_rgb.ply", pcd_rgb)
    print(f"RGB point cloud exported to {filepath_base}_rgb.ply")
    
    # Segment colored PLY
    export_segmented_point_cloud_ply(points, colors, segments, f"{filepath_base}_segmented.ply")
    
    # Save labels as numpy file
    np.savez(f"{filepath_base}_labels.npz", 
             points=points, 
             colors=colors, 
             segments=segments,
             class_names=ADE20K_CLASSES)
    print(f"Labels saved to {filepath_base}_labels.npz")


def visualize_point_cloud_open3d(points, colors=None, window_name="Point Cloud"):
    """Display 3D point cloud with Open3D viewer"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_segmented_point_cloud(points, segments, window_name="Segmented Point Cloud"):
    """Display 3D point cloud colored by segments"""
    segment_colors = get_segment_colors(int(segments.max()) + 1)
    colored_by_segment = segment_colors[segments.astype(int)]
    
    visualize_point_cloud_open3d(points, colored_by_segment, window_name)


def print_segment_statistics(segments):
    """Print statistics about detected segments"""
    unique, counts = np.unique(segments, return_counts=True)
    print("\n" + "="*50)
    print("SEGMENT STATISTICS")
    print("="*50)
    
    # Sort by count (descending)
    sorted_indices = np.argsort(-counts)
    
    print(f"{'Class ID':<10} {'Class Name':<25} {'Points':<15} {'Percentage':<10}")
    print("-"*60)
    
    total_points = len(segments)
    for idx in sorted_indices[:15]:  # Top 15 classes
        class_id = unique[idx]
        count = counts[idx]
        percentage = (count / total_points) * 100
        class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else f"unknown_{class_id}"
        print(f"{class_id:<10} {class_name:<25} {count:<15} {percentage:.2f}%")
    
    if len(unique) > 15:
        print(f"... and {len(unique) - 15} more classes")
    print("="*50 + "\n")


def process_pipeline_with_segmentation(data_folder, conf_thresh=0.4, visualize=True, clean=True):
    """Main pipeline to generate segmented point cloud from images"""
    print(f"Starting segmented pipeline for dataset: {data_folder}")
    print("="*60)
    
    # Setup paths
    paths = setup_paths(data_folder)
    print(f"Output directory: {paths['results']}")
    
    # Find images
    image_files = load_images_from_folder(paths['data'])
    if not image_files:
        print("No images found. Exiting.")
        return
    
    # Run segmentation FIRST (before loading DA3 to save memory)
    print("\n[1/5] Loading Mask2Former segmentation model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, seg_processor, _ = load_segmentation_model()
    print("Segmentation model loaded.")
    
    print("\n[2/5] Running segmentation on images...")
    seg_maps = []
    for i, img_path in enumerate(image_files):
        print(f"  Segmenting image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        seg_map = run_segmentation(seg_model, seg_processor, img_path, device)
        seg_maps.append(seg_map)
        torch.cuda.empty_cache()  # Clear cache after each image
    print("Segmentation complete.")
    
    # Free segmentation model memory completely
    del seg_model
    del seg_processor
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("Segmentation model unloaded to free memory.")
    
    # Now load depth model
    print("\n[3/5] Loading Depth Anything 3 model...")
    da3_model, device = load_da3_model()
    print("Depth model loaded.")
    
    # Run depth inference
    print("\n[4/5] Running depth estimation...")
    prediction = run_da3_inference(da3_model, image_files)
    
    # Generate point cloud with segments
    print("\n[5/5] Generating segmented point cloud...")
    print(f"Using confidence threshold: {conf_thresh}")
    points_3d, colors_3d, segments = merge_point_clouds_with_segments(
        prediction, seg_maps, conf_thresh=conf_thresh
    )
    
    # Clean point cloud (optional)
    if clean:
        print("\nCleaning point cloud...")
        final_pts, final_cols, final_segs = clean_point_cloud_with_segments(
            points_3d, colors_3d, segments
        )
        print(f"Cleaned point cloud: {len(final_pts)} points (removed {len(points_3d) - len(final_pts)} outliers)")
    else:
        print("\nSkipping point cloud cleaning.")
        final_pts, final_cols, final_segs = points_3d, colors_3d, segments
    
    # Print segment statistics
    print_segment_statistics(final_segs)
    
    # Visualize
    if visualize:
        print("Visualizing RGB point cloud...")
        visualize_point_cloud_open3d(final_pts, final_cols, window_name="RGB Point Cloud")
        
        print("Visualizing segmented point cloud...")
        visualize_segmented_point_cloud(final_pts, final_segs, window_name="Segmented Point Cloud")
    
    # Export results
    output_base = os.path.join(paths['results'], "scene_pointcloud")
    export_point_cloud_with_labels(final_pts, final_cols, final_segs, output_base)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return final_pts, final_cols, final_segs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate segmented 3D point cloud from images using Depth Anything 3 + Mask2Former"
    )
    parser.add_argument("--data_folder", type=str, default="SAMPLE_SCENE", 
                        help="Name of the folder in DATA directory")
    parser.add_argument("--conf_thresh", type=float, default=0.4, 
                        help="Confidence threshold for filtering points")
    parser.add_argument("--no_visualize", action="store_true", 
                        help="Disable visualization")
    parser.add_argument("--no_clean", action="store_true", 
                        help="Disable point cloud cleaning (outlier removal)")
    
    args = parser.parse_args()
    
    process_pipeline_with_segmentation(
        args.data_folder, 
        args.conf_thresh, 
        visualize=not args.no_visualize,
        clean=not args.no_clean
    )
