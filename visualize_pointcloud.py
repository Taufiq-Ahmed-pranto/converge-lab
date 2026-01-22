import open3d as o3d
import os
import argparse

def setup_paths(data_folder="SAMPLE_SCENE", base_path=None):
    """Create project paths for data, results, and masks"""
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    paths = {
        'results': os.path.join(base_path, 'RESULTS', data_folder),
        'ply_file': os.path.join(base_path, 'RESULTS', data_folder, "scene_pointcloud.ply")
    }
    return paths

def visualize_point_cloud(data_folder="SAMPLE_SCENE"):
    paths = setup_paths(data_folder)
    ply_path = paths['ply_file']
    
    if not os.path.exists(ply_path):
        print(f"Error: Point cloud file not found at {ply_path}")
        print(f"Make sure you have run generate_pointcloud.py for '{data_folder}' first.")
        return

    print(f"Loading point cloud from {ply_path}...")
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Failed to read point cloud: {e}")
        return

    if not pcd.has_points():
        print("Error: The loaded point cloud is empty!")
        return

    print(f"Successfully loaded {len(pcd.points)} points.")
    print("\nControls:")
    print("  - Left click + drag: Rotate")
    print("  - Ctrl + Left click + drag: Pan")
    print("  - Wheel: Zoom")
    print("  - +/-: Increase/decrease point size")
    print("  - N: Toggle point normal rendering")
    print("  - Q: Quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {data_folder}", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Add a coordinate frame for reference (size 1.0)
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]  # Dark grey background
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated 3D point cloud")
    parser.add_argument("--data_folder", type=str, default="SAMPLE_SCENE", help="Name of the folder in DATA directory to visualize")
    
    args = parser.parse_args()
    
    visualize_point_cloud(args.data_folder)
