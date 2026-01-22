#!/usr/bin/env python3
"""
Depth Maps and Confidence Maps Explanation & Visualization
Educational script to understand depth estimation concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path
import open3d as o3d

class DepthMapExplainer:
    def __init__(self, output_dir="/home/sasan/Desktop/project/depth_anything/output"):
        self.output_dir = Path(output_dir)
        print("🎓 Depth Map & Confidence Map Educational Tool")
        
    def load_data(self, image_idx=0):
        """Load all data for a specific image"""
        try:
            # Load original image
            original_path = self.output_dir / f"original_{image_idx:03d}.png"
            original = cv2.imread(str(original_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Load depth map (from Gaussian Splat data)
            gsplat_path = self.output_dir / f"gaussian_splat_{image_idx:03d}.npz"
            gsplat_data = np.load(gsplat_path)
            
            # Reconstruct depth map from 3D points
            positions = gsplat_data['positions']
            depth_values = positions[:, 2]  # Z-coordinate is depth
            
            # Load confidence map (from Gaussian Splat data)
            opacities = gsplat_data['opacities'].flatten()
            
            # Load statistics
            stats_path = self.output_dir / f"stats_{image_idx:03d}.json"
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            return {
                'original': original,
                'depth_values': depth_values,
                'confidence_values': opacities,
                'positions': positions,
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def explain_depth_maps(self, data):
        """Explain what depth maps are and how they work"""
        print("\n" + "="*60)
        print("📏 DEPTH MAPS EXPLANATION")
        print("="*60)
        
        print("""
🎯 What is a Depth Map?
A depth map is a 2D image where each pixel contains the distance from the camera 
to the object at that pixel location. Instead of RGB color values, each pixel 
stores a depth value (usually in meters).

🔍 Key Concepts:
• CLOSER objects have SMALLER depth values (dark in visualization)
• FARTHER objects have LARGER depth values (bright in visualization)  
• Each pixel represents: "How far is this point from the camera?"

🌈 Color Coding (typical):
• PURPLE/DARK BLUE = Very close (small depth values)
• RED/ORANGE = Medium distance  
• YELLOW/WHITE = Far away (large depth values)
""")
        
        stats = data['stats']
        depth_values = data['depth_values']
        
        print(f"📊 Your Depth Map Statistics:")
        print(f"• Minimum depth: {stats['min']:.3f}m (closest point)")
        print(f"• Maximum depth: {stats['max']:.3f}m (farthest point)")  
        print(f"• Average depth: {stats['mean']:.3f}m")
        print(f"• Depth range: {stats['max'] - stats['min']:.3f}m")
        print(f"• Total points: {len(depth_values):,}")
        
        # Depth distribution analysis
        print(f"\n📈 Depth Distribution:")
        close_points = np.sum(depth_values < 1.0)
        medium_points = np.sum((depth_values >= 1.0) & (depth_values < 2.5))
        far_points = np.sum(depth_values >= 2.5)
        
        total_points = len(depth_values)
        print(f"• Close objects (<1.0m): {close_points:,} ({close_points/total_points*100:.1f}%)")
        print(f"• Medium distance (1.0-2.5m): {medium_points:,} ({medium_points/total_points*100:.1f}%)")
        print(f"• Far objects (>2.5m): {far_points:,} ({far_points/total_points*100:.1f}%)")
    
    def explain_confidence_maps(self, data):
        """Explain what confidence maps are and how they work"""
        print("\n" + "="*60)
        print("🎯 CONFIDENCE MAPS EXPLANATION") 
        print("="*60)
        
        print("""
🛡️ What is a Confidence Map?
A confidence map tells us HOW RELIABLE each depth prediction is. It's like a 
"trust score" for each pixel's depth value. Some areas are easier to estimate 
depth for than others.

🔍 Key Concepts:
• HIGH confidence = The AI is very sure about this depth value
• LOW confidence = The AI is uncertain about this depth value
• Confidence ranges from 0.0 (no confidence) to 1.0 (maximum confidence)

🌈 Color Coding (typical):
• DARK/BLACK = Low confidence (uncertain depth)
• BRIGHT/YELLOW = High confidence (reliable depth)

🤔 What affects confidence?
• Texture: Textured areas = higher confidence
• Edges: Sharp edges = higher confidence  
• Smooth areas: Plain surfaces = lower confidence
• Lighting: Well-lit areas = higher confidence
• Occlusions: Hidden areas = lower confidence
""")
        
        confidence_values = data['confidence_values']
        
        print(f"📊 Your Confidence Map Statistics:")
        print(f"• Average confidence: {np.mean(confidence_values):.3f}")
        print(f"• Minimum confidence: {np.min(confidence_values):.3f}")
        print(f"• Maximum confidence: {np.max(confidence_values):.3f}")
        print(f"• Confidence std dev: {np.std(confidence_values):.3f}")
        
        # Confidence distribution
        print(f"\n📈 Confidence Distribution:")
        low_conf = np.sum(confidence_values < 0.3)
        med_conf = np.sum((confidence_values >= 0.3) & (confidence_values < 0.7))
        high_conf = np.sum(confidence_values >= 0.7)
        
        total = len(confidence_values)
        print(f"• Low confidence (<0.3): {low_conf:,} ({low_conf/total*100:.1f}%)")
        print(f"• Medium confidence (0.3-0.7): {med_conf:,} ({med_conf/total*100:.1f}%)")
        print(f"• High confidence (>0.7): {high_conf:,} ({high_conf/total*100:.1f}%)")
    
    def create_educational_visualization(self, data, save_path="depth_confidence_explanation.png"):
        """Create comprehensive educational visualization"""
        
        original = data['original']
        positions = data['positions']
        depth_values = data['depth_values']
        confidence_values = data['confidence_values']
        
        # Reshape depth and confidence back to image dimensions
        H, W = original.shape[:2]
        
        # Create depth and confidence images
        depth_2d = np.zeros((H, W))
        confidence_2d = np.zeros((H, W))
        
        # Fill the 2D arrays (simple approach - in practice this needs proper mapping)
        # For demonstration, we'll create synthetic 2D versions
        x_coords = (np.arange(len(positions)) % W).astype(int)
        y_coords = (np.arange(len(positions)) // W).astype(int)
        
        valid_mask = (y_coords < H) & (x_coords < W)
        y_coords = y_coords[valid_mask]
        x_coords = x_coords[valid_mask]
        depth_vals = depth_values[valid_mask]
        conf_vals = confidence_values[valid_mask]
        
        depth_2d[y_coords, x_coords] = depth_vals
        confidence_2d[y_coords, x_coords] = conf_vals
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Depth Maps & Confidence Maps - Complete Explanation', fontsize=16, fontweight='bold')
        
        # Row 1: Original data
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('📷 Original RGB Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        depth_vis = axes[0, 1].imshow(depth_2d, cmap='plasma', vmin=depth_values.min(), vmax=depth_values.max())
        axes[0, 1].set_title('📏 Depth Map\n(Purple=Close, Yellow=Far)', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1], shrink=0.8, label='Distance (meters)')
        
        conf_vis = axes[0, 2].imshow(confidence_2d, cmap='viridis', vmin=0, vmax=1)
        axes[0, 2].set_title('🎯 Confidence Map\n(Dark=Uncertain, Bright=Confident)', fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(conf_vis, ax=axes[0, 2], shrink=0.8, label='Confidence (0-1)')
        
        # Row 2: Analysis plots
        # Depth histogram
        axes[1, 0].hist(depth_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('📊 Depth Distribution')
        axes[1, 0].set_xlabel('Depth (meters)')
        axes[1, 0].set_ylabel('Number of Points')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence histogram  
        axes[1, 1].hist(confidence_values, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('📊 Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Number of Points')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Depth vs Confidence scatter
        scatter = axes[1, 2].scatter(depth_values, confidence_values, c=confidence_values, 
                                   cmap='viridis', alpha=0.6, s=1)
        axes[1, 2].set_title('🔍 Depth vs Confidence\nRelationship')
        axes[1, 2].set_xlabel('Depth (meters)')
        axes[1, 2].set_ylabel('Confidence Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Educational examples
        # Show depth slices
        close_mask = depth_values < np.percentile(depth_values, 33)
        medium_mask = (depth_values >= np.percentile(depth_values, 33)) & (depth_values < np.percentile(depth_values, 67))
        far_mask = depth_values >= np.percentile(depth_values, 67)
        
        depth_slices = np.zeros_like(depth_2d)
        depth_slices[y_coords[close_mask], x_coords[close_mask]] = 1  # Close objects
        depth_slices[y_coords[medium_mask], x_coords[medium_mask]] = 2  # Medium objects  
        depth_slices[y_coords[far_mask], x_coords[far_mask]] = 3  # Far objects
        
        slice_vis = axes[2, 0].imshow(depth_slices, cmap='RdYlBu_r')
        axes[2, 0].set_title('📍 Depth Layers\n(Blue=Close, Red=Far)')
        axes[2, 0].axis('off')
        
        # Show confidence regions
        high_conf_mask = confidence_values > 0.7
        med_conf_mask = (confidence_values >= 0.3) & (confidence_values <= 0.7)
        low_conf_mask = confidence_values < 0.3
        
        conf_regions = np.zeros_like(confidence_2d)
        conf_regions[y_coords[high_conf_mask], x_coords[high_conf_mask]] = 3  # High confidence
        conf_regions[y_coords[med_conf_mask], x_coords[med_conf_mask]] = 2   # Medium confidence
        conf_regions[y_coords[low_conf_mask], x_coords[low_conf_mask]] = 1   # Low confidence
        
        region_vis = axes[2, 1].imshow(conf_regions, cmap='RdYlGn')
        axes[2, 1].set_title('🛡️ Confidence Regions\n(Red=Low, Green=High)')
        axes[2, 1].axis('off')
        
        # Statistics summary
        stats_text = f"""
📊 Key Statistics:
        
Depth Range: {data['stats']['min']:.2f}m - {data['stats']['max']:.2f}m
Average Depth: {data['stats']['mean']:.2f}m
        
Confidence Stats:
• Mean: {np.mean(confidence_values):.3f}
• Min: {np.min(confidence_values):.3f}  
• Max: {np.max(confidence_values):.3f}

Point Distribution:
• Close (<1m): {np.sum(depth_values < 1.0):,}
• Medium (1-2.5m): {np.sum((depth_values >= 1.0) & (depth_values < 2.5)):,}
• Far (>2.5m): {np.sum(depth_values >= 2.5):,}
        
Quality Indicators:
• High Conf: {np.sum(confidence_values > 0.7):,} pts
• Med Conf: {np.sum((confidence_values >= 0.3) & (confidence_values <= 0.7)):,} pts  
• Low Conf: {np.sum(confidence_values < 0.3):,} pts
"""
        
        axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('📈 Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Educational visualization saved: {save_path}")
    
    def explain_how_it_works(self):
        """Explain the technical process of how depth estimation works"""
        print("\n" + "="*60)
        print("⚙️ HOW DEPTH ESTIMATION WORKS")
        print("="*60)
        
        print("""
🧠 Deep Learning Process:

1️⃣ INPUT PROCESSING:
   • Take a single RGB image (monocular)
   • Resize to standard dimensions (e.g., 280×504)
   • Normalize pixel values

2️⃣ NEURAL NETWORK ARCHITECTURE:
   • Encoder: Extracts features from the image
   • Decoder: Converts features to depth predictions
   • Multi-scale processing for different detail levels

3️⃣ TRAINING DATA:
   • Millions of image-depth pairs
   • RGB-D cameras, LiDAR scans, stereo cameras
   • Synthetic datasets (games, simulations)

4️⃣ DEPTH PREDICTION:
   • Each pixel gets a depth value
   • Network learns depth cues:
     - Object size (smaller = farther)
     - Perspective (parallel lines converge)
     - Occlusion (which objects are in front)
     - Atmospheric perspective (hazy = farther)
     - Shadows and lighting

5️⃣ CONFIDENCE ESTIMATION:
   • Network also predicts uncertainty
   • Based on feature quality and consistency
   • Higher confidence for clear, textured areas
   • Lower confidence for ambiguous regions

🎯 Depth Anything 3 Improvements:
   • Better handling of diverse scenes
   • More accurate metric depth
   • Improved confidence estimation
   • Faster inference speed
""")

    def practical_applications(self):
        """Explain practical applications of depth maps"""
        print("\n" + "="*60)
        print("🌟 PRACTICAL APPLICATIONS")
        print("="*60)
        
        print("""
🚗 AUTONOMOUS VEHICLES:
   • Obstacle detection and avoidance
   • Lane keeping and navigation
   • Parking assistance

📱 SMARTPHONE FEATURES:
   • Portrait mode (background blur)
   • AR applications (object placement)
   • 3D scanning and modeling

🎮 GAMING & VR:
   • Real-time 3D reconstruction
   • Gesture recognition
   • Immersive environments

🏭 ROBOTICS:
   • Navigation and path planning
   • Grasping and manipulation
   • Quality inspection

🎬 FILM & MEDIA:
   • 3D content creation
   • Special effects
   • Virtual production

🏠 SMART HOME:
   • Security systems
   • Occupancy detection
   • Automated lighting

🔬 RESEARCH:
   • 3D scene understanding
   • Medical imaging
   • Archaeological documentation
""")

def main():
    """Main educational program"""
    print("🎓 Welcome to Depth Map & Confidence Map Learning!")
    print("This tool will help you understand depth estimation concepts.")
    
    explainer = DepthMapExplainer()
    
    # Check if we have data to work with
    if not explainer.output_dir.exists():
        print("❌ Output directory not found!")
        print("Please run 'python gaussian_splatting_test.py' first to generate data.")
        return
    
    # Find available data
    gaussian_files = list(explainer.output_dir.glob("gaussian_splat_*.npz"))
    if not gaussian_files:
        print("❌ No Gaussian Splat data found!")
        print("Please run 'python gaussian_splatting_test.py' first to generate data.")
        return
    
    print(f"✅ Found {len(gaussian_files)} datasets to analyze")
    
    # Load and analyze first dataset
    data = explainer.load_data(0)
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Educational explanations
    explainer.explain_depth_maps(data)
    explainer.explain_confidence_maps(data)
    explainer.explain_how_it_works()
    explainer.practical_applications()
    
    # Create comprehensive visualization
    print("\n🎨 Creating educational visualization...")
    explainer.create_educational_visualization(data)
    
    print("\n" + "="*60)
    print("🎉 CONGRATULATIONS!")
    print("You now understand depth maps and confidence maps!")
    print("="*60)
    print("""
🔑 Key Takeaways:
• Depth maps store distance information for each pixel
• Confidence maps show how reliable each depth estimate is
• Together they enable robust 3D understanding
• Used in many real-world applications

🚀 Next Steps:
• Try the Open3D viewer: python open3d_gaussian_viewer.py
• Experiment with different images
• Explore 3D applications
""")

if __name__ == "__main__":
    main()