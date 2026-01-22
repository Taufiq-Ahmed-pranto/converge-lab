import glob
import os
import torch
import numpy as np
import open3d as o3d
import argparse
from depth_anything_3.api import DepthAnything3


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


def depth_to_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points using camera parameters"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter by confidence if provided
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, rgb_image = u[valid_mask], v[valid_mask], depth_map[valid_mask], rgb_image[valid_mask]
    else:
        u, v, depth_map = u.flatten(), v.flatten(), depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3)
    
    # Back-project to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates using extrinsics (w2c format)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R  # Inverse transform
    
    colors = rgb_image.astype(np.float32) / 255.0
    
    return points_world, colors


def merge_point_clouds(prediction, conf_thresh=0.5):
    """Combine all frames into single point cloud"""
    all_points = []
    all_colors = []
    
    n_frames = len(prediction.depth)
    
    for i in range(n_frames):
        points, colors = depth_to_point_cloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            conf_thresh
        )
        all_points.append(points)
        all_colors.append(colors)
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"Merged point cloud: {len(merged_points)} points")
    return merged_points, merged_colors


def clean_point_cloud_open3d(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """Cleans a point cloud using Statistical Outlier Removal (SOR) via Open3D."""
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

    return cleaned_points, cleaned_colors


def export_point_cloud_ply(points, colors, filepath):
    """Export point cloud to PLY format"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Point cloud exported to {filepath}")


def visualize_point_cloud_open3d(points, colors=None, window_name="Point Cloud"):
    """Display 3D point cloud with Open3D viewer"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def process_pipeline(data_folder, conf_thresh=0.4, visualize=True, clean=True):
    """Main pipeline to generate cleaned point cloud from images"""
    print(f"Starting pipeline for dataset: {data_folder}")
    
    # Setup paths
    paths = setup_paths(data_folder)
    print(f"Output directory: {paths['results']}")
    
    # Find images
    image_files = load_images_from_folder(paths['data'])
    if not image_files:
        print("No images found. Exiting.")
        return
        
    # Load model
    print("Loading model...")
    model, device = load_da3_model()
    print("Model loaded.")
    
    # Run inference
    print("Running inference...")
    prediction = run_da3_inference(model, image_files)
    
    # Generate point cloud
    print(f"Generating point cloud with confidence threshold: {conf_thresh}")
    points_3d, colors_3d = merge_point_clouds(prediction, conf_thresh=conf_thresh)
    
    # Clean point cloud (optional)
    if clean:
        print("Cleaning point cloud...")
        clean_pts, clean_cols = clean_point_cloud_open3d(points_3d, colors_3d)
        print(f"Cleaned point cloud: {len(clean_pts)} points (removed {len(points_3d) - len(clean_pts)} outliers)")
        final_pts, final_cols = clean_pts, clean_cols
    else:
        print("Skipping point cloud cleaning.")
        final_pts, final_cols = points_3d, colors_3d
    
    # Visualize point cloud
    if visualize:
        print("Visualizing point cloud...")
        visualize_point_cloud_open3d(final_pts, final_cols, window_name="Point Cloud")
    
    # Export result
    ply_path = os.path.join(paths['results'], "scene_pointcloud.ply")
    export_point_cloud_ply(final_pts, final_cols, ply_path)
    
    print("Pipeline completed successfully!")
    return final_pts, final_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from images using Depth Anything 3")
    parser.add_argument("--data_folder", type=str, default="SAMPLE_SCENE", help="Name of the folder in DATA directory")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold for filtering points")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--no_clean", action="store_true", help="Disable point cloud cleaning (outlier removal)")
    
    args = parser.parse_args()
    
    process_pipeline(args.data_folder, args.conf_thresh, visualize=not args.no_visualize, clean=not args.no_clean)
