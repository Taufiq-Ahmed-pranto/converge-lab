import glob, os, torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from depth_anything_3.api import DepthAnything3

# Test if gsplat is working
print("🔍 Testing Gaussian Splatting support...")
try:
    import gsplat
    print(f"✅ gsplat installed successfully! Version: {gsplat.__version__}")
    GSPLAT_AVAILABLE = True
except ImportError as e:
    print(f"❌ gsplat not available: {e}")
    GSPLAT_AVAILABLE = False

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
model = model.to(device=device)
example_path = "/home/sasan/Desktop/project/depth_anything/Depth-Anything-3/assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(images)

print("Shape Information:")
print(f"Processed images: {prediction.processed_images.shape}")
print(f"Depth maps: {prediction.depth.shape}")  
print(f"Confidence maps: {prediction.conf.shape}")  
print(f"Extrinsics: {prediction.extrinsics.shape}")
print(f"Intrinsics: {prediction.intrinsics.shape}")

# Save individual depth maps and create point clouds
output_dir = "/home/sasan/Desktop/project/depth_anything/output"
os.makedirs(output_dir, exist_ok=True)

print("\n🎯 Generating 3D Point Clouds and Gaussian Splats...")

for i in range(len(prediction.depth)):
    print(f"\n--- Processing Image {i+1} ---")
    
    # Get data for this image
    rgb = prediction.processed_images[i]  # [H, W, 3]
    depth = prediction.depth[i]  # [H, W]
    confidence = prediction.conf[i]  # [H, W]
    K = prediction.intrinsics[i]  # [3, 3] camera intrinsics
    
    # Create 3D point cloud from depth
    H, W = depth.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert to 3D coordinates using camera intrinsics
    fx, fy = K[0, 0], K[1, 1]  # focal lengths
    cx, cy = K[0, 2], K[1, 2]  # principal point
    
    # Convert depth to 3D points
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into point cloud [N, 3]
    points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    colors = rgb.reshape(-1, 3) / 255.0  # normalize colors
    
    # Filter out invalid points (confidence-based filtering)
    confidence_threshold = 0.5
    valid_mask = confidence.flatten() > confidence_threshold
    
    points_3d_filtered = points_3d[valid_mask]
    colors_filtered = colors[valid_mask]
    
    # Save point cloud as PLY file
    ply_path = os.path.join(output_dir, f"pointcloud_{i:03d}.ply")
    
    # Write PLY header
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d_filtered)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for j in range(len(points_3d_filtered)):
            x, y, z = points_3d_filtered[j]
            r, g, b = (colors_filtered[j] * 255).astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"✅ Point cloud saved: {ply_path}")
    print(f"   Total points: {len(points_3d):,}")
    print(f"   Filtered points: {len(points_3d_filtered):,} (confidence > {confidence_threshold})")
    
    # If gsplat is available, create Gaussian Splat
    if GSPLAT_AVAILABLE:
        print(f"🎨 Creating Gaussian Splat representation...")
        
        # Convert to torch tensors
        points_torch = torch.from_numpy(points_3d_filtered).float().to(device)
        colors_torch = torch.from_numpy(colors_filtered).float().to(device)
        
        # Create simple Gaussian parameters
        # In a real scenario, you'd optimize these parameters
        num_points = len(points_3d_filtered)
        
        # Initialize Gaussian parameters
        scales = torch.ones(num_points, 3, device=device) * 0.01  # Small initial scale
        rotations = torch.zeros(num_points, 4, device=device)  # Identity quaternion
        rotations[:, 0] = 1.0  # w component of identity quaternion
        
        opacities = torch.ones(num_points, 1, device=device) * 0.8  # Semi-transparent
        
        # Save Gaussian Splat data
        gsplat_data = {
            'positions': points_torch.cpu().numpy(),
            'colors': colors_torch.cpu().numpy(),
            'scales': scales.cpu().numpy(),
            'rotations': rotations.cpu().numpy(),
            'opacities': opacities.cpu().numpy(),
        }
        
        gsplat_path = os.path.join(output_dir, f"gaussian_splat_{i:03d}.npz")
        np.savez(gsplat_path, **gsplat_data)
        
        print(f"✅ Gaussian Splat saved: {gsplat_path}")
        
        # Create a simple visualization using gsplat
        try:
            # This is a basic example - in practice you'd need proper camera setup
            print(f"🖼️  Rendering Gaussian Splat preview...")
            
            # Simple rendering setup (this is just for demonstration)
            render_data = {
                'means': points_torch,
                'scales': scales, 
                'quats': rotations,
                'colors': colors_torch,
                'opacities': opacities
            }
            
            # Save render parameters for later use
            render_path = os.path.join(output_dir, f"render_params_{i:03d}.pt")
            torch.save(render_data, render_path)
            print(f"✅ Render parameters saved: {render_path}")
            
        except Exception as e:
            print(f"⚠️  Rendering preview failed: {e}")
    
    # Create depth statistics
    depth_stats = {
        'min': float(depth.min()),
        'max': float(depth.max()),
        'mean': float(depth.mean()),
        'std': float(depth.std()),
        'valid_pixels': int(np.sum(confidence > confidence_threshold))
    }
    
    stats_path = os.path.join(output_dir, f"stats_{i:03d}.json")
    import json
    with open(stats_path, 'w') as f:
        json.dump(depth_stats, f, indent=2)
    
    print(f"📊 Depth stats: min={depth_stats['min']:.3f}m, max={depth_stats['max']:.3f}m, mean={depth_stats['mean']:.3f}m")

print(f"\n🎉 3D Processing Complete!")
print(f"📁 All outputs saved to: {output_dir}")
print(f"\n📋 Generated files per image:")
print(f"   • pointcloud_XXX.ply - 3D point cloud")
if GSPLAT_AVAILABLE:
    print(f"   • gaussian_splat_XXX.npz - Gaussian Splat data")
    print(f"   • render_params_XXX.pt - Rendering parameters")
print(f"   • stats_XXX.json - Depth statistics")
print(f"\n💡 You can view PLY files in:")
print(f"   • MeshLab (free)")
print(f"   • CloudCompare (free)")
print(f"   • Blender (free)")
print(f"   • Open3D viewer")

# Create a simple Open3D visualization if available
try:
    import open3d as o3d
    print(f"\n🔍 Creating Open3D visualization...")
    
    # Load and display the first point cloud
    if len(images) > 0:
        ply_file = os.path.join(output_dir, "pointcloud_000.ply")
        if os.path.exists(ply_file):
            pcd = o3d.io.read_point_cloud(ply_file)
            print(f"✅ Loaded point cloud with {len(pcd.points)} points")
            
            # Save a screenshot
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            screenshot_path = os.path.join(output_dir, "pointcloud_preview.png") 
            vis.capture_screen_image(screenshot_path)
            vis.destroy_window()
            print(f"📸 Point cloud preview saved: {screenshot_path}")
            
except ImportError:
    print(f"ℹ️  Open3D not available for automatic visualization")
except Exception as e:
    print(f"⚠️  Open3D visualization failed: {e}")

print(f"\n🚀 Ready for Gaussian Splatting workflows!")