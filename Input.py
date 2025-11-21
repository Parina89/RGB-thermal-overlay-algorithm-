import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ThermalImageOverlay:
    """
    A PyTorch-based thermal and RGB image overlay tool.
    Image 1: Thermal image (base layer)
    Image 2: RGB image (overlay layer)
    """
    
    def __init__(self, device=None):
        """Initialize with specified device (cuda/cpu)."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.image1 = None  # Thermal image
        self.image2 = None  # RGB image
    
    def load_image(self, path, image_name="image"):
        """Load an image and convert to PyTorch tensor."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and convert to tensor (C, H, W)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).to(self.device)
        print(f"{image_name} loaded: {img.shape[1]}x{img.shape[0]} pixels")
        return tensor
    
    def load_images(self, image1_path, image2_path):
        """
        Load both images.
        Image 1: Thermal image (base)
        Image 2: RGB image (overlay)
        """
        self.image1 = self.load_image(image1_path, "Image 1 (Thermal)")
        self.image2 = self.load_image(image2_path, "Image 2 (RGB)")
        self._resize_to_match()
    
    def _resize_to_match(self):
        """Resize images to match dimensions."""
        if self.image1 is None or self.image2 is None:
            return
        
        _, h1, w1 = self.image1.shape
        _, h2, w2 = self.image2.shape
        
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        if h1 != target_h or w1 != target_w:
            self.image1 = torch.nn.functional.interpolate(
                self.image1.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        if h2 != target_h or w2 != target_w:
            self.image2 = torch.nn.functional.interpolate(
                self.image2.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        print(f"Images resized to: {target_w}x{target_h} pixels")
    
    def blend_normal(self, opacity=0.5):
        """Normal blending: Image 1 + Image 2 with opacity."""
        return self.image1 * (1 - opacity) + self.image2 * opacity
    
    def blend_multiply(self, opacity=0.5):
        """Multiply blend mode."""
        blended = self.image1 * self.image2
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_screen(self, opacity=0.5):
        """Screen blend mode."""
        blended = 1 - (1 - self.image1) * (1 - self.image2)
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_overlay(self, opacity=0.5):
        """Overlay blend mode."""
        mask = self.image1 < 0.5
        blended = torch.where(
            mask,
            2 * self.image1 * self.image2,
            1 - 2 * (1 - self.image1) * (1 - self.image2)
        )
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_difference(self, opacity=0.5):
        """Difference blend mode."""
        blended = torch.abs(self.image1 - self.image2)
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_add(self, opacity=0.5):
        """Additive blend mode."""
        blended = torch.clamp(self.image1 + self.image2, 0, 1)
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_subtract(self, opacity=0.5):
        """Subtract blend mode."""
        blended = torch.clamp(self.image1 - self.image2, 0, 1)
        return self.image1 * (1 - opacity) + blended * opacity
    
    def blend_soft_light(self, opacity=0.5):
        """Soft light blend mode."""
        mask = self.image2 < 0.5
        blended = torch.where(
            mask,
            self.image1 - (1 - 2 * self.image2) * self.image1 * (1 - self.image1),
            self.image1 + (2 * self.image2 - 1) * (torch.sqrt(self.image1) - self.image1)
        )
        return self.image1 * (1 - opacity) + blended * opacity
    
    def overlay(self, blend_mode='normal', opacity=0.5):
        """
        Overlay Image 2 onto Image 1 with specified blend mode and opacity.
        
        Args:
            blend_mode: One of 'normal', 'multiply', 'screen', 'overlay', 
                       'difference', 'add', 'subtract', 'soft_light'
            opacity: Float between 0 and 1 (0 = only Image 1, 1 = full blend)
        
        Returns:
            PyTorch tensor of the blended image
        """
        if self.image1 is None or self.image2 is None:
            raise ValueError("Please load both images first using load_images()")
        
        blend_functions = {
            'normal': self.blend_normal,
            'multiply': self.blend_multiply,
            'screen': self.blend_screen,
            'overlay': self.blend_overlay,
            'difference': self.blend_difference,
            'add': self.blend_add,
            'subtract': self.blend_subtract,
            'soft_light': self.blend_soft_light
        }
        
        if blend_mode not in blend_functions:
            raise ValueError(f"Unknown blend mode: {blend_mode}. Available: {list(blend_functions.keys())}")
        
        result = blend_functions[blend_mode](opacity)
        return torch.clamp(result, 0, 1)
    
    def tensor_to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array for display."""
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    def display_result(self, result, title="Overlay Result"):
        """Display the blended result using matplotlib."""
        plt.figure(figsize=(12, 8))
        plt.imshow(self.tensor_to_numpy(result))
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def display_comparison(self, result, blend_mode, opacity):
        """Display Image 1, Image 2, and result side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(self.tensor_to_numpy(self.image1))
        axes[0].set_title('Image 1 (Thermal)')
        axes[0].axis('off')
        
        axes[1].imshow(self.tensor_to_numpy(self.image2))
        axes[1].set_title('Image 2 (RGB)')
        axes[1].axis('off')
        
        axes[2].imshow(self.tensor_to_numpy(result))
        axes[2].set_title(f'Overlay ({blend_mode}, {int(opacity*100)}%)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_all_blend_modes(self, opacity=0.5):
        """Display all blend modes for comparison."""
        modes = ['normal', 'multiply', 'screen', 'overlay', 
                 'difference', 'add', 'subtract', 'soft_light']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for ax, mode in zip(axes, modes):
            result = self.overlay(mode, opacity)
            ax.imshow(self.tensor_to_numpy(result))
            ax.set_title(f'{mode.capitalize()}')
            ax.axis('off')
        
        plt.suptitle(f'All Blend Modes (Opacity: {int(opacity*100)}%)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def save_result(self, result, output_path):
        """Save the blended result to a file."""
        img_np = (self.tensor_to_numpy(result) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)
        print(f"Result saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the overlay tool
    overlay_tool = ThermalImageOverlay()
    
    # Load images
    # Image 1: Thermal image (base layer)
    # Image 2: RGB image (overlay layer)
    image1_path = "image1.jpg"  # Replace with your thermal image path
    image2_path = "image2.jpg"  # Replace with your RGB image path
    
    try:
        overlay_tool.load_images(image1_path, image2_path)
        
        # Create overlay with different blend modes
        result = overlay_tool.overlay(blend_mode='overlay', opacity=0.5)
        
        # Display comparison
        overlay_tool.display_comparison(result, 'overlay', 0.5)
        
        # Display all blend modes
        overlay_tool.display_all_blend_modes(opacity=0.5)
        
        # Save result
        overlay_tool.save_result(result, "overlay_result.jpg")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update image1_path and image2_path with your actual image files.")
