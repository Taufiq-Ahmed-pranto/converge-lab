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


def create_point_cloud_per_frame(prediction, conf_thresh=0.5):
    """Create separate point clouds for each frame"""
    point_clouds = []
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
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        point_clouds.append(pcd)
        
    return point_clouds


def refine_alignment_icp(point_clouds, voxel_size=0.02, max_correspondence_distance=0.05):
    """
    Refine alignment between point clouds using ICP (Iterative Closest Point).
    This corrects small misalignments from estimated camera poses.
    """
    if len(point_clouds) < 2:
        return point_clouds
    
    print(f"\nRefining alignment with ICP (voxel_size={voxel_size})...")
    
    refined_pcds = [point_clouds[0]]  # First cloud is reference
    
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]
        target = refined_pcds[-1]  # Align to previous cloud
        
        # Downsample for faster ICP
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Estimate normals for point-to-plane ICP
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Apply transformation to original (non-downsampled) point cloud
        source_refined = source.transform(result.transformation)
        refined_pcds.append(source_refined)
        
        print(f"  Frame {i}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
    
    return refined_pcds


def refine_alignment_multiway(point_clouds, voxel_size=0.01):
    """
    Global multiway registration for better alignment of all point clouds.
    Uses pose graph optimization for global consistency.
    """
    if len(point_clouds) < 2:
        return point_clouds
    
    print(f"\nPerforming multiway registration (voxel_size={voxel_size})...")
    
    # Downsample all point clouds
    pcds_down = []
    for pcd in point_clouds:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        pcds_down.append(pcd_down)
    
    # Build pose graph
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    n_pcds = len(pcds_down)
    max_correspondence_distance = voxel_size * 1.5
    
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, min(source_id + 3, n_pcds)):  # Connect nearby frames
            # Pairwise registration
            result = o3d.pipelines.registration.registration_icp(
                pcds_down[source_id], pcds_down[target_id],
                max_correspondence_distance,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            
            if target_id == source_id + 1:  # Odometry edge
                odometry = result.transformation @ odometry
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            
            # Add edge to pose graph
            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds_down[source_id], pcds_down[target_id],
                max_correspondence_distance,
                result.transformation
            )
            
            uncertain = target_id != source_id + 1
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, result.transformation, information, uncertain
                )
            )
    
    # Optimize pose graph
    print("  Optimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    
    # Apply optimized transformations
    refined_pcds = []
    for i, pcd in enumerate(point_clouds):
        pcd_transformed = pcd.transform(pose_graph.nodes[i].pose)
        refined_pcds.append(pcd_transformed)
    
    print("  Multiway registration complete.")
    return refined_pcds


def merge_point_clouds_simple(point_clouds):
    """Simple merge by concatenating all point clouds"""
    merged = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        merged += pcd
    return merged


def voxel_downsample_and_denoise(pcd, voxel_size=0.01):
    """
    Voxel downsampling averages points within each voxel,
    which reduces noise and creates more uniform point density.
    """
    print(f"Voxel downsampling with size {voxel_size}...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"  Points: {len(pcd.points)} → {len(pcd_down.points)}")
    return pcd_down


def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers"""
    print(f"Removing outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  Points: {len(pcd.points)} → {len(pcd_clean.points)} (removed {len(pcd.points) - len(pcd_clean.points)})")
    return pcd_clean


def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    """Remove points with few neighbors in radius"""
    print(f"Radius outlier removal (points={nb_points}, radius={radius})...")
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  Points: {len(pcd.points)} → {len(pcd_clean.points)}")
    return pcd_clean


def filter_by_multiview_consistency(point_clouds, distance_threshold=0.03, min_views=2):
    """
    Keep only points that are observed from multiple views.
    Points seen from multiple cameras are more reliable.
    """
    print(f"\nFiltering by multi-view consistency (min_views={min_views})...")
    
    if len(point_clouds) < 2:
        return merge_point_clouds_simple(point_clouds)
    
    # Merge all points with view indices
    all_points = []
    all_colors = []
    view_indices = []
    
    for i, pcd in enumerate(point_clouds):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        all_points.append(points)
        all_colors.append(colors)
        view_indices.extend([i] * len(points))
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    view_indices = np.array(view_indices)
    
    # Build KD-tree for finding nearby points
    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(all_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_merged)
    
    # For each point, count how many different views have nearby points
    consistent_mask = np.zeros(len(all_points), dtype=bool)
    
    for i in range(len(all_points)):
        [k, idx, _] = kdtree.search_radius_vector_3d(all_points[i], distance_threshold)
        if k > 0:
            nearby_views = set(view_indices[idx])
            if len(nearby_views) >= min_views:
                consistent_mask[i] = True
    
    # Filter points
    consistent_points = all_points[consistent_mask]
    consistent_colors = all_colors[consistent_mask]
    
    print(f"  Kept {len(consistent_points)} / {len(all_points)} points ({100*len(consistent_points)/len(all_points):.1f}%)")
    
    pcd_consistent = o3d.geometry.PointCloud()
    pcd_consistent.points = o3d.utility.Vector3dVector(consistent_points)
    pcd_consistent.colors = o3d.utility.Vector3dVector(consistent_colors)
    
    return pcd_consistent


def export_point_cloud_ply_o3d(pcd, filepath):
    """Export Open3D point cloud to PLY format"""
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Point cloud exported to {filepath}")


def visualize_point_cloud_o3d(pcd, window_name="Point Cloud"):
    """Display Open3D point cloud"""
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def process_pipeline(data_folder, conf_thresh=0.4, visualize=True, clean=True, 
                     use_icp=False, use_multiway=False, use_multiview_filter=False,
                     voxel_size=0.0):
    """
    Main pipeline to generate point cloud from images.
    
    Args:
        data_folder: Name of folder in DATA directory
        conf_thresh: Confidence threshold for depth filtering
        visualize: Show interactive visualization
        clean: Apply statistical outlier removal
        use_icp: Use ICP for pairwise alignment refinement
        use_multiway: Use multiway registration for global alignment
        use_multiview_filter: Keep only points seen from multiple views
        voxel_size: Voxel size for downsampling (0 = no downsampling, keeps all points)
    """
    print(f"Starting pipeline for dataset: {data_folder}")
    print("="*60)
    
    # Setup paths
    paths = setup_paths(data_folder)
    print(f"Output directory: {paths['results']}")
    
    # Find images
    image_files = load_images_from_folder(paths['data'])
    if not image_files:
        print("No images found. Exiting.")
        return
        
    # Load model
    print("\nLoading model...")
    model, device = load_da3_model()
    print("Model loaded.")
    
    # Run inference
    print("\nRunning inference...")
    prediction = run_da3_inference(model, image_files)
    
    # Create per-frame point clouds
    print(f"\nGenerating point clouds with confidence threshold: {conf_thresh}")
    point_clouds = create_point_cloud_per_frame(prediction, conf_thresh=conf_thresh)
    print(f"Created {len(point_clouds)} point clouds")
    
    # === ALIGNMENT REFINEMENT ===
    # Note: ICP/multiway uses internal downsampling for speed, but applies
    # the computed transformations to the FULL point clouds
    if use_multiway:
        # Multiway registration (best for global consistency)
        # Use a reasonable voxel size for registration computation only
        registration_voxel_size = 0.02 if voxel_size <= 0 else voxel_size * 2
        point_clouds = refine_alignment_multiway(point_clouds, voxel_size=registration_voxel_size)
    elif use_icp:
        # Pairwise ICP refinement
        registration_voxel_size = 0.02 if voxel_size <= 0 else voxel_size * 2
        point_clouds = refine_alignment_icp(point_clouds, voxel_size=registration_voxel_size)
    
    # === MERGE POINT CLOUDS ===
    if use_multiview_filter:
        # Keep only points seen from multiple views
        distance_thresh = 0.03 if voxel_size <= 0 else voxel_size * 3
        merged_pcd = filter_by_multiview_consistency(point_clouds, 
                                                      distance_threshold=distance_thresh,
                                                      min_views=2)
    else:
        # Simple merge
        merged_pcd = merge_point_clouds_simple(point_clouds)
    
    print(f"\nMerged point cloud: {len(merged_pcd.points)} points")
    
    # === POST-PROCESSING ===
    # Voxel downsampling (only if voxel_size > 0)
    if voxel_size > 0:
        merged_pcd = voxel_downsample_and_denoise(merged_pcd, voxel_size=voxel_size)
    else:
        print("Skipping voxel downsampling (keeping all points)")
    
    # Statistical outlier removal
    if clean:
        merged_pcd = statistical_outlier_removal(merged_pcd, nb_neighbors=20, std_ratio=2.0)
    
    # Get final points and colors
    final_pts = np.asarray(merged_pcd.points)
    final_cols = np.asarray(merged_pcd.colors)
    
    print(f"\nFinal point cloud: {len(final_pts)} points")
    
    # Visualize
    if visualize:
        print("\nVisualizing point cloud...")
        visualize_point_cloud_o3d(merged_pcd, window_name="Point Cloud")
    
    # Export result
    ply_path = os.path.join(paths['results'], "scene_pointcloud.ply")
    export_point_cloud_ply_o3d(merged_pcd, ply_path)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return final_pts, final_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from images using Depth Anything 3")
    parser.add_argument("--data_folder", type=str, default="SAMPLE_SCENE", 
                        help="Name of the folder in DATA directory")
    parser.add_argument("--conf_thresh", type=float, default=0.4, 
                        help="Confidence threshold for filtering points (0.0-1.0)")
    parser.add_argument("--voxel_size", type=float, default=0.0, 
                        help="Voxel size for downsampling (0 = no downsampling, keeps all points)")
    parser.add_argument("--no_visualize", action="store_true", 
                        help="Disable visualization")
    parser.add_argument("--no_clean", action="store_true", 
                        help="Disable point cloud cleaning (outlier removal)")
    parser.add_argument("--icp", action="store_true", 
                        help="Use ICP for pairwise alignment refinement")
    parser.add_argument("--multiway", action="store_true", 
                        help="Use multiway registration for global alignment (recommended)")
    parser.add_argument("--multiview", action="store_true", 
                        help="Filter to keep only points seen from multiple views")
    
    args = parser.parse_args()
    
    process_pipeline(
        args.data_folder, 
        args.conf_thresh, 
        visualize=not args.no_visualize, 
        clean=not args.no_clean,
        use_icp=args.icp,
        use_multiway=args.multiway,
        use_multiview_filter=args.multiview,
        voxel_size=args.voxel_size
    )
