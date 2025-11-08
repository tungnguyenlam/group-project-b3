import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from data_pipeline.data_loader_factory import DataLoaderFactory

def visualize_batch(images, targets, split_name):
    """
    Visualizes a batch of images with their bounding boxes and masks.
    """
    batch_size = len(images)
    
    # Create a figure with subplots based on the batch size
    ncols = 2
    nrows = (batch_size + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10 * nrows))
    
    # Handle single image case
    if batch_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(batch_size):
        # Convert the tensor image to a NumPy array for visualization
        image = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Clip values to [0, 1] range for proper display
        image = np.clip(image, 0, 1)
        
        # Get the corresponding targets
        target = targets[i]
        boxes = target['boxes'].cpu().numpy()
        masks = target['masks'].cpu().numpy()
        
        ax = axes[i]
        
        # Display the base image
        ax.imshow(image, cmap='gray' if image.shape[2] == 1 else None)
        ax.set_title(f"Sample from {split_name} split\nImage: {target['file_name']}\nBoxes: {len(boxes)}, Masks: {len(masks)}")
        ax.axis('off')

        print(f"\nImage {i}: {target['file_name']}")
        print(f"  - Image shape: {image.shape}")
        print(f"  - Num boxes: {len(boxes)}")
        print(f"  - Num masks: {len(masks)}")
        print(f"  - Masks shape: {masks.shape if len(masks) > 0 else 'empty'}")

        # Create a combined mask overlay with different colors for each instance
        if len(masks) > 0:
            h, w = masks.shape[1], masks.shape[2]
            combined_mask = np.zeros((h, w, 4))  # RGBA
            
            for idx, mask in enumerate(masks):
                # Generate a random bright color for each mask
                color = np.array([
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0), 
                    random.uniform(0.3, 1.0),
                    0.4  # Alpha transparency
                ])
                
                # Apply the color to mask regions
                mask_bool = mask > 0
                combined_mask[mask_bool] = color
                
                print(f"  - Mask {idx}: {mask_bool.sum()} pixels")
            
            # Overlay the combined mask
            ax.imshow(combined_mask, interpolation='nearest')

        # Overlay bounding boxes
        for idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Random color for each box
            edge_color = (random.random(), random.random(), random.random())
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=3, 
                edgecolor=edge_color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x_min, y_min - 5, 
                f'#{idx+1}', 
                color='white', 
                fontsize=12, 
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=edge_color, alpha=0.7)
            )
            
    # Hide any unused subplots
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    print("\n" + "="*70)


if __name__ == '__main__':
    # --- Configuration ---
    JSON_DIR = 'MangaSegmentation/jsons_processed'
    IMAGE_ROOT = 'Manga109_released_2023_12_07/Manga109_released_2023_12_07/images'
    BATCH_SIZE = 4
    
    print("="*70)
    print("Initializing DataLoaderFactory...")
    print("="*70)
    data_loader_factory = DataLoaderFactory(json_dir=JSON_DIR, image_root=IMAGE_ROOT)

    # --- Visualize Training Data ---
    print("\nLoading a batch from the 'train' split for visualization...")
    train_loader = data_loader_factory.create_data_loader(
        split='train', 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4 
    )
    
    train_iterator = iter(train_loader)
    num_train_batches = 3

    for i in range(num_train_batches):
        train_images, train_targets = next(train_iterator)
        print(f"\nLoaded batch {i+1} of {len(train_images)} images from the training set.")
        visualize_batch(train_images, train_targets, f'train_batch_{i+1}')
    
    print(f"\nLoaded a batch of {len(train_images)} images from the training set.")
    print(f"Image tensor shape: {train_images.shape}")
    visualize_batch(train_images, train_targets, 'train')

    # --- Visualize Validation Data ---
    print("\nLoading a batch from the 'validation' split for visualization...")
    val_loader = data_loader_factory.create_data_loader(
        split='validation', 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )
    
    val_iterator = iter(val_loader)
    num_val_batches = 2

    for i in range(num_val_batches):
        val_images, val_targets = next(val_iterator)
        print(f"\nLoaded batch {i+1} of {len(val_images)} images from the validation set.")
        visualize_batch(val_images, val_targets, f'validation_batch_{i+1}')
    
    print(f"\nLoaded a batch of {len(val_images)} images from the validation set.")
    print(f"Image tensor shape: {val_images.shape}")
    visualize_batch(val_images, val_targets, 'validation')