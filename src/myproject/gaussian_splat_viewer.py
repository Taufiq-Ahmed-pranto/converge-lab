import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox
import json

class GaussianSplatViewer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_data = None
        self.current_stats = None
        self.fig = None
        self.ax = None
        self.scatter = None
        
        # Camera parameters
        self.azimuth = 45
        self.elevation = 30
        self.distance = 5.0
        self.point_size = 1.0
        self.opacity_scale = 1.0
        
        print(f"🎮 Gaussian Splat Viewer initialized on {self.device}")
    
    def load_gaussian_splat(self, file_path):
        """Load Gaussian Splat data from npz file"""
        try:
            print(f"📂 Loading Gaussian Splat: {file_path}")
            data = np.load(file_path)
            
            self.current_data = {
                'positions': data['positions'],
                'colors': data['colors'], 
                'scales': data['scales'],
                'rotations': data['rotations'],
                'opacities': data['opacities']
            }
            
            # Load corresponding stats if available
            stats_file = file_path.replace('gaussian_splat_', 'stats_').replace('.npz', '.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.current_stats = json.load(f)
            
            print(f"✅ Loaded {len(self.current_data['positions'])} Gaussian points")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Gaussian Splat: {e}")
            return False
    
    def create_interactive_viewer(self):
        """Create interactive matplotlib viewer"""
        if self.current_data is None:
            print("❌ No Gaussian Splat data loaded!")
            return
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(221, projection='3d')
        
        # Additional 2D projections
        ax_xy = self.fig.add_subplot(222)  # XY projection
        ax_xz = self.fig.add_subplot(223)  # XZ projection  
        ax_yz = self.fig.add_subplot(224)  # YZ projection
        
        positions = self.current_data['positions']
        colors = self.current_data['colors']
        opacities = self.current_data['opacities'].flatten()
        
        # Apply opacity scaling
        alpha_values = opacities * self.opacity_scale
        alpha_values = np.clip(alpha_values, 0, 1)
        
        # Create RGBA colors
        rgba_colors = np.column_stack([colors, alpha_values])
        
        # Main 3D scatter plot
        self.scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=rgba_colors, s=self.point_size, alpha=0.6
        )
        
        # Set up 3D plot
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('3D Gaussian Splat View')
        
        # 2D projections
        ax_xy.scatter(positions[:, 0], positions[:, 1], c=colors, s=0.5, alpha=0.6)
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('XY Projection (Top View)')
        ax_xy.set_aspect('equal')
        
        ax_xz.scatter(positions[:, 0], positions[:, 2], c=colors, s=0.5, alpha=0.6)
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title('XZ Projection (Front View)')
        ax_xz.set_aspect('equal')
        
        ax_yz.scatter(positions[:, 1], positions[:, 2], c=colors, s=0.5, alpha=0.6)
        ax_yz.set_xlabel('Y (m)')
        ax_yz.set_ylabel('Z (m)')
        ax_yz.set_title('YZ Projection (Side View)')
        ax_yz.set_aspect('equal')
        
        # Add control sliders
        self.add_interactive_controls()
        
        # Add statistics display
        if self.current_stats:
            stats_text = f"""Statistics:
Depth Range: {self.current_stats['min']:.2f}m - {self.current_stats['max']:.2f}m
Mean Depth: {self.current_stats['mean']:.2f}m ± {self.current_stats['std']:.2f}m
Valid Points: {self.current_stats['valid_pixels']:,}
Total Gaussians: {len(positions):,}"""
            
            self.fig.text(0.02, 0.02, stats_text, fontsize=9, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Set initial camera view
        self.update_camera_view()
        
        plt.tight_layout()
        plt.show()
    
    def add_interactive_controls(self):
        """Add interactive sliders and buttons"""
        # Make space for controls
        plt.subplots_adjust(bottom=0.25)
        
        # Azimuth slider
        ax_azimuth = plt.axes([0.1, 0.15, 0.3, 0.03])
        self.slider_azimuth = Slider(ax_azimuth, 'Azimuth', 0, 360, 
                                   valinit=self.azimuth, valfmt='%0.0f°')
        self.slider_azimuth.on_changed(self.update_azimuth)
        
        # Elevation slider  
        ax_elevation = plt.axes([0.1, 0.11, 0.3, 0.03])
        self.slider_elevation = Slider(ax_elevation, 'Elevation', -90, 90,
                                     valinit=self.elevation, valfmt='%0.0f°')
        self.slider_elevation.on_changed(self.update_elevation)
        
        # Distance slider
        ax_distance = plt.axes([0.1, 0.07, 0.3, 0.03])
        self.slider_distance = Slider(ax_distance, 'Distance', 1, 20,
                                    valinit=self.distance, valfmt='%0.1f')
        self.slider_distance.on_changed(self.update_distance)
        
        # Point size slider
        ax_size = plt.axes([0.5, 0.15, 0.3, 0.03])
        self.slider_size = Slider(ax_size, 'Point Size', 0.1, 10,
                                valinit=self.point_size, valfmt='%0.1f')
        self.slider_size.on_changed(self.update_point_size)
        
        # Opacity slider
        ax_opacity = plt.axes([0.5, 0.11, 0.3, 0.03])
        self.slider_opacity = Slider(ax_opacity, 'Opacity', 0.1, 2.0,
                                   valinit=self.opacity_scale, valfmt='%0.1f')
        self.slider_opacity.on_changed(self.update_opacity)
        
        # Reset button
        ax_reset = plt.axes([0.5, 0.07, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset View')
        self.button_reset.on_clicked(self.reset_view)
        
        # Save view button
        ax_save = plt.axes([0.7, 0.07, 0.1, 0.04])
        self.button_save = Button(ax_save, 'Save View')
        self.button_save.on_clicked(self.save_view)
    
    def update_azimuth(self, val):
        self.azimuth = val
        self.update_camera_view()
    
    def update_elevation(self, val):
        self.elevation = val
        self.update_camera_view()
    
    def update_distance(self, val):
        self.distance = val
        self.update_camera_view()
    
    def update_point_size(self, val):
        self.point_size = val
        if self.scatter:
            self.scatter.set_sizes([self.point_size] * len(self.current_data['positions']))
            self.fig.canvas.draw_idle()
    
    def update_opacity(self, val):
        self.opacity_scale = val
        if self.scatter and self.current_data:
            positions = self.current_data['positions']
            colors = self.current_data['colors']
            opacities = self.current_data['opacities'].flatten()
            
            # Apply new opacity scaling
            alpha_values = opacities * self.opacity_scale
            alpha_values = np.clip(alpha_values, 0, 1)
            rgba_colors = np.column_stack([colors, alpha_values])
            
            self.scatter.set_color(rgba_colors)
            self.fig.canvas.draw_idle()
    
    def update_camera_view(self):
        """Update 3D camera view"""
        if self.ax:
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            self.ax.dist = self.distance
            self.fig.canvas.draw_idle()
    
    def reset_view(self, event):
        """Reset to default view"""
        self.azimuth = 45
        self.elevation = 30
        self.distance = 5.0
        self.point_size = 1.0
        self.opacity_scale = 1.0
        
        # Update sliders
        self.slider_azimuth.reset()
        self.slider_elevation.reset()
        self.slider_distance.reset()
        self.slider_size.reset() 
        self.slider_opacity.reset()
        
        self.update_camera_view()
    
    def save_view(self, event):
        """Save current view as image"""
        if self.fig:
            timestamp = str(int(np.random.random() * 10000))
            filename = f"/home/sasan/Desktop/project/depth_anything/output/gaussian_view_{timestamp}.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 View saved as: {filename}")

class GaussianSplatApp:
    def __init__(self):
        self.viewer = GaussianSplatViewer()
        self.output_dir = "/home/sasan/Desktop/project/depth_anything/output"
        
    def run_gui(self):
        """Run GUI file selector"""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        print("🎮 Gaussian Splat Viewer - Interactive Mode")
        print("📁 Select a Gaussian Splat file (.npz) to view...")
        
        # Get available files
        npz_files = [f for f in os.listdir(self.output_dir) 
                    if f.startswith('gaussian_splat_') and f.endswith('.npz')]
        
        if not npz_files:
            messagebox.showerror("Error", "No Gaussian Splat files found in output directory!")
            return
        
        print(f"📋 Available files: {npz_files}")
        
        # File selection dialog
        file_path = filedialog.askopenfilename(
            title="Select Gaussian Splat File",
            initialdir=self.output_dir,
            filetypes=[("Gaussian Splat files", "*.npz"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.viewer.load_gaussian_splat(file_path):
                print("🎨 Creating interactive viewer...")
                self.viewer.create_interactive_viewer()
        else:
            print("❌ No file selected")
    
    def run_batch_viewer(self):
        """View all Gaussian Splats in sequence"""
        npz_files = sorted([f for f in os.listdir(self.output_dir) 
                           if f.startswith('gaussian_splat_') and f.endswith('.npz')])
        
        if not npz_files:
            print("❌ No Gaussian Splat files found!")
            return
        
        print(f"🎬 Batch viewing {len(npz_files)} Gaussian Splats...")
        
        for i, filename in enumerate(npz_files):
            file_path = os.path.join(self.output_dir, filename)
            print(f"\n📺 Viewing {i+1}/{len(npz_files)}: {filename}")
            
            if self.viewer.load_gaussian_splat(file_path):
                self.viewer.create_interactive_viewer()
                
                if i < len(npz_files) - 1:
                    input("Press Enter to view next Gaussian Splat...")

# Advanced Gaussian Splat Renderer using gsplat
class GSPlatRenderer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 Advanced GSplat Renderer initialized on {self.device}")
    
    def render_gaussian_splat(self, npz_path, output_size=(800, 600)):
        """Render Gaussian Splat using gsplat library"""
        try:
            import gsplat
            
            # Load data
            data = np.load(npz_path)
            
            # Convert to torch tensors
            means = torch.from_numpy(data['positions']).float().to(self.device)
            colors = torch.from_numpy(data['colors']).float().to(self.device)  
            scales = torch.from_numpy(data['scales']).float().to(self.device)
            quats = torch.from_numpy(data['rotations']).float().to(self.device)
            opacities = torch.from_numpy(data['opacities']).float().to(self.device)
            
            print(f"🎨 Rendering {len(means)} Gaussians...")
            
            # Simple camera setup (you can modify this)
            height, width = output_size
            
            # Camera intrinsics (simple pinhole camera)
            fx = fy = width * 0.7  # Rough focal length
            cx, cy = width / 2, height / 2
            
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy], 
                [0, 0, 1]
            ]).float().to(self.device)
            
            # Camera extrinsics (identity for now)
            viewmat = torch.eye(4).float().to(self.device)
            
            # Render using gsplat
            print("🖼️  Performing Gaussian Splatting render...")
            
            # This is a simplified example - real gsplat rendering requires more setup
            rendered_data = {
                'means': means,
                'colors': colors,
                'scales': scales,
                'quats': quats,
                'opacities': opacities,
                'K': K,
                'viewmat': viewmat
            }
            
            # Save render data for external renderers
            render_output = npz_path.replace('.npz', '_render_ready.pt')
            torch.save(rendered_data, render_output)
            print(f"✅ Render-ready data saved: {render_output}")
            
            return rendered_data
            
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            return None

def main():
    """Main function with menu options"""
    print("🎮 Gaussian Splat Viewer")
    print("=" * 40)
    print("1. Interactive GUI Viewer")
    print("2. Batch View All Splats") 
    print("3. Advanced GSplat Renderer")
    print("4. Exit")
    
    app = GaussianSplatApp()
    renderer = GSPlatRenderer()
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                app.run_gui()
            elif choice == "2":
                app.run_batch_viewer()
            elif choice == "3":
                output_dir = "/home/sasan/Desktop/project/depth_anything/output"
                npz_files = [f for f in os.listdir(output_dir) 
                            if f.startswith('gaussian_splat_') and f.endswith('.npz')]
                
                if npz_files:
                    print("Available Gaussian Splat files:")
                    for i, f in enumerate(npz_files):
                        print(f"  {i+1}. {f}")
                    
                    try:
                        idx = int(input(f"Select file (1-{len(npz_files)}): ")) - 1
                        if 0 <= idx < len(npz_files):
                            file_path = os.path.join(output_dir, npz_files[idx])
                            renderer.render_gaussian_splat(file_path)
                    except (ValueError, IndexError):
                        print("❌ Invalid selection")
                else:
                    print("❌ No Gaussian Splat files found!")
                    
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()