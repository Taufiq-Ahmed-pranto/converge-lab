import glob, os, torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from depth_anything_3.api import DepthAnything3

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

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Depth Anything 3 - Results Visualization', fontsize=16)

for i in range(2):  # For each image
    # Original image
    original_img = prediction.processed_images[i]  # [H, W, 3]
    axes[i, 0].imshow(original_img)
    axes[i, 0].set_title(f'Original Image {i+1}')
    axes[i, 0].axis('off')
    
    # Depth map
    depth_map = prediction.depth[i]  # [H, W]
    depth_vis = axes[i, 1].imshow(depth_map, cmap='plasma')
    axes[i, 1].set_title(f'Depth Map {i+1}')
    axes[i, 1].axis('off')
    plt.colorbar(depth_vis, ax=axes[i, 1], shrink=0.8)
    
    # Confidence map
    conf_map = prediction.conf[i]  # [H, W]
    conf_vis = axes[i, 2].imshow(conf_map, cmap='viridis')
    axes[i, 2].set_title(f'Confidence Map {i+1}')
    axes[i, 2].axis('off')
    plt.colorbar(conf_vis, ax=axes[i, 2], shrink=0.8)

plt.tight_layout()
plt.savefig('/home/sasan/Desktop/project/depth_anything/depth_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save individual depth maps as images
output_dir = "/home/sasan/Desktop/project/depth_anything/output"
os.makedirs(output_dir, exist_ok=True)

for i in range(len(prediction.depth)):
    # Save original processed image
    original_path = os.path.join(output_dir, f"original_{i:03d}.png")
    cv2.imwrite(original_path, cv2.cvtColor(prediction.processed_images[i], cv2.COLOR_RGB2BGR))
    
    # Save depth map as colorized image
    depth_normalized = (prediction.depth[i] - prediction.depth[i].min()) / (prediction.depth[i].max() - prediction.depth[i].min())
    depth_colored = plt.cm.plasma(depth_normalized)[:, :, :3] * 255
    depth_path = os.path.join(output_dir, f"depth_{i:03d}.png")
    cv2.imwrite(depth_path, cv2.cvtColor(depth_colored.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Save confidence map
    conf_normalized = (prediction.conf[i] - prediction.conf[i].min()) / (prediction.conf[i].max() - prediction.conf[i].min())
    conf_colored = plt.cm.viridis(conf_normalized)[:, :, :3] * 255
    conf_path = os.path.join(output_dir, f"confidence_{i:03d}.png")
    cv2.imwrite(conf_path, cv2.cvtColor(conf_colored.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print(f"✅ Saved results for image {i+1}:")
    print(f"   Original: {original_path}")
    print(f"   Depth: {depth_path}")
    print(f"   Confidence: {conf_path}")

# Print depth statistics
print(f"\n📊 Depth Statistics:")
for i in range(len(prediction.depth)):
    depth = prediction.depth[i]
    print(f"Image {i+1}: min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")

print(f"\n✅ Visualization saved as: /home/sasan/Desktop/project/depth_anything/depth_results.png")
print(f"✅ Individual images saved in: {output_dir}")