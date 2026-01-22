#!/usr/bin/env python3
"""
Open3D Gaussian Splat Viewer
Interactive 3D visualization of Gaussian Splats using Open3D
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import json
from pathlib import Path
import time

class Open3DGaussianViewer:
    def __init__(self):
        self.vis = None
        self.point_cloud = None
        self.current_data = None
        self.output_dir = Path("/home/sasan/Desktop/project/depth_anything/output")
        
        # Visualization parameters
        self.point_size = 3.0
        self.show_coordinate_frame = True
        self.background_color = [0.1, 0.1, 0.1]  # Dark background
        
        print("🎮 Open3D Gaussian Splat Viewer initialized")
    
    def load_gaussian_splat(self, file_path):
        """Load Gaussian Splat data from npz file"""
        try:
            print(f"📂 Loading Gaussian Splat: {file_path}")
            
            # Load npz file
            data = np.load(file_path)
            
            self.current_data = {
                'positions': data['positions'],
                'colors': data['colors'],
                'scales': data['scales'],
                'rotations': data['rotations'],
                'opacities': data['opacities']
            }
            
            # Load corresponding stats if available
            stats_file = str(file_path).replace('gaussian_splat_', 'stats_').replace('.npz', '.json')
            stats_info = {}
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats_info = json.load(f)
            
            print(f"✅ Loaded {len(self.current_data['positions'])} Gaussian points")
            if stats_info:
                print(f"📊 Depth range: {stats_info.get('min', 0):.2f}m - {stats_info.get('max', 0):.2f}m")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Gaussian Splat: {e}")
            return False
    
    def load_point_cloud(self, ply_path):
        """Load point cloud from PLY file"""
        try:
            print(f"📂 Loading point cloud: {ply_path}")
            pcd = o3d.io.read_point_cloud(str(ply_path))
            
            if len(pcd.points) == 0:
                print("❌ Point cloud is empty!")
                return None
            
            print(f"✅ Loaded point cloud with {len(pcd.points)} points")
            return pcd
            
        except Exception as e:
            print(f"❌ Failed to load point cloud: {e}")
            return None
    
    def gaussian_to_point_cloud(self):
        """Convert Gaussian Splat data to Open3D point cloud"""
        if self.current_data is None:
            return None
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Set positions
        pcd.points = o3d.utility.Vector3dVector(self.current_data['positions'])
        
        # Set colors (apply opacity)
        colors = self.current_data['colors']
        opacities = self.current_data['opacities'].flatten()
        
        # Apply opacity to colors (simple approach)
        alpha_weighted_colors = colors * opacities[:, np.newaxis]
        alpha_weighted_colors = np.clip(alpha_weighted_colors, 0, 1)
        
        pcd.colors = o3d.utility.Vector3dVector(alpha_weighted_colors)
        
        return pcd
    
    def create_enhanced_visualization(self, pcd):
        """Create enhanced visualization with additional features"""
        geometries = [pcd]
        
        # Add coordinate frame
        if self.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            geometries.append(coord_frame)
        
        # Estimate normals for better visualization
        if len(pcd.points) > 100:
            print("🔄 Computing normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        
        # Create bounding box
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Red bounding box
        geometries.append(bbox)
        
        return geometries
    
    def launch_interactive_viewer(self, geometries, window_name="Gaussian Splat Viewer"):
        """Launch Open3D interactive viewer"""
        print("🚀 Launching interactive viewer...")
        
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=1200, height=800)
        
        # Add geometries
        for geom in geometries:
            self.vis.add_geometry(geom)
        
        # Configure rendering options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.point_size = self.point_size
        render_option.show_coordinate_frame = self.show_coordinate_frame
        
        # Set up camera
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_up([0.0, -1.0, 0.0])
        
        print("🎮 Interactive controls:")
        print("  • Mouse: Rotate, zoom, pan")
        print("  • Mouse wheel: Zoom in/out")
        print("  • Left click + drag: Rotate")
        print("  • Right click + drag: Pan")
        print("  • Middle click + drag: Zoom")
        print("  • Close window to exit")
        
        # Save initial screenshot
        timestamp = int(time.time())
        screenshot_path = f"gaussian_view_{timestamp}.png"
        
        # Run visualizer (simplified version without key callbacks)
        try:
            # Update visualization and run
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Capture initial screenshot
            self.vis.capture_screen_image(screenshot_path)
            print(f"📸 Initial view saved as: {screenshot_path}")
            
            # Run the interactive viewer
            self.vis.run()
            
        except Exception as e:
            print(f"⚠️ Viewer error: {e}")
        finally:
            self.vis.destroy_window()
    
    def view_gaussian_splat(self, file_path):
        """Main function to view a Gaussian Splat"""
        if not self.load_gaussian_splat(file_path):
            return False
        
        # Convert to point cloud
        print("🔄 Converting Gaussian Splat to point cloud...")
        pcd = self.gaussian_to_point_cloud()
        
        if pcd is None:
            print("❌ Failed to create point cloud")
            return False
        
        # Create enhanced visualization
        geometries = self.create_enhanced_visualization(pcd)
        
        # Launch viewer
        self.launch_interactive_viewer(geometries, f"Gaussian Splat: {Path(file_path).name}")
        
        return True
    
    def view_point_cloud_file(self, ply_path):
        """View a PLY point cloud file directly"""
        pcd = self.load_point_cloud(ply_path)
        
        if pcd is None:
            return False
        
        # Create enhanced visualization
        geometries = self.create_enhanced_visualization(pcd)
        
        # Launch viewer
        self.launch_interactive_viewer(geometries, f"Point Cloud: {Path(ply_path).name}")
        
        return True
    
    def compare_multiple_views(self, file_paths):
        """Compare multiple Gaussian Splats or point clouds"""
        print(f"🔄 Loading {len(file_paths)} files for comparison...")
        
        all_geometries = []
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]
        
        for i, file_path in enumerate(file_paths):
            if str(file_path).endswith('.npz'):
                if self.load_gaussian_splat(file_path):
                    pcd = self.gaussian_to_point_cloud()
                    if pcd is not None:
                        # Apply unique color for comparison
                        color = colors[i % len(colors)]
                        pcd.paint_uniform_color(color)
                        
                        # Offset position for side-by-side comparison
                        offset = np.array([i * 3.0, 0, 0])
                        pcd.translate(offset)
                        
                        all_geometries.append(pcd)
                        print(f"✅ Loaded Gaussian Splat {i+1}: {Path(file_path).name}")
            
            elif str(file_path).endswith('.ply'):
                pcd = self.load_point_cloud(file_path)
                if pcd is not None:
                    # Apply unique color for comparison
                    color = colors[i % len(colors)]
                    pcd.paint_uniform_color(color)
                    
                    # Offset position for side-by-side comparison
                    offset = np.array([i * 3.0, 0, 0])
                    pcd.translate(offset)
                    
                    all_geometries.append(pcd)
                    print(f"✅ Loaded Point Cloud {i+1}: {Path(file_path).name}")
        
        if not all_geometries:
            print("❌ No valid geometries loaded!")
            return False
        
        # Add coordinate frame
        if self.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            all_geometries.append(coord_frame)
        
        # Launch comparison viewer
        self.launch_interactive_viewer(all_geometries, "Gaussian Splat Comparison")
        
        return True
    
    def list_available_files(self):
        """List all available Gaussian Splat and point cloud files"""
        print("📁 Available files in output directory:")
        print(f"   Directory: {self.output_dir}")
        
        gaussian_files = list(self.output_dir.glob("gaussian_splat_*.npz"))
        ply_files = list(self.output_dir.glob("pointcloud_*.ply"))
        
        if gaussian_files:
            print("\n🎨 Gaussian Splat files (.npz):")
            for i, f in enumerate(gaussian_files):
                print(f"   {i+1:2d}. {f.name}")
        
        if ply_files:
            print("\n📊 Point Cloud files (.ply):")
            for i, f in enumerate(ply_files):
                print(f"   {i+1:2d}. {f.name}")
        
        if not gaussian_files and not ply_files:
            print("   ❌ No files found!")
            return [], []
        
        return gaussian_files, ply_files

def main():
    parser = argparse.ArgumentParser(description="Open3D Gaussian Splat Viewer")
    parser.add_argument("--file", "-f", type=str, help="Path to Gaussian Splat (.npz) or Point Cloud (.ply) file")
    parser.add_argument("--compare", "-c", nargs="+", help="Compare multiple files")
    parser.add_argument("--list", "-l", action="store_true", help="List available files")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive file selection")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size for visualization")
    
    args = parser.parse_args()
    
    viewer = Open3DGaussianViewer()
    viewer.point_size = args.point_size
    
    if args.list or args.interactive:
        gaussian_files, ply_files = viewer.list_available_files()
        
        if not args.interactive:
            return
        
        # Interactive selection
        all_files = gaussian_files + ply_files
        if not all_files:
            print("❌ No files available for viewing")
            return
        
        print(f"\n🎮 Interactive Mode")
        print("Select a file to view:")
        for i, f in enumerate(all_files):
            file_type = "Gaussian Splat" if f.suffix == ".npz" else "Point Cloud"
            print(f"   {i+1:2d}. {f.name} ({file_type})")
        
        try:
            choice = int(input(f"\nEnter choice (1-{len(all_files)}): ")) - 1
            if 0 <= choice < len(all_files):
                selected_file = all_files[choice]
                print(f"🎯 Selected: {selected_file.name}")
                
                if selected_file.suffix == ".npz":
                    viewer.view_gaussian_splat(selected_file)
                else:
                    viewer.view_point_cloud_file(selected_file)
            else:
                print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("👋 Goodbye!")
            return
    
    elif args.compare:
        file_paths = [Path(f) for f in args.compare]
        viewer.compare_multiple_views(file_paths)
    
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        if file_path.suffix == ".npz":
            viewer.view_gaussian_splat(file_path)
        elif file_path.suffix == ".ply":
            viewer.view_point_cloud_file(file_path)
        else:
            print(f"❌ Unsupported file format: {file_path.suffix}")
    
    else:
        # Default: show first available file
        gaussian_files, ply_files = viewer.list_available_files()
        
        if gaussian_files:
            print(f"\n🎯 Viewing first Gaussian Splat: {gaussian_files[0].name}")
            viewer.view_gaussian_splat(gaussian_files[0])
        elif ply_files:
            print(f"\n🎯 Viewing first Point Cloud: {ply_files[0].name}")
            viewer.view_point_cloud_file(ply_files[0])
        else:
            print("❌ No files available. Run gaussian_splatting_test.py first!")

if __name__ == "__main__":
    main()