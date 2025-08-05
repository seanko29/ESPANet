import torch
import torch.nn.functional as F

def degradation_operator(hr, scale):
    """
    Downsample a high-resolution tensor to low-resolution using bicubic interpolation.
    
    Args:
        hr (torch.Tensor): High-resolution image tensor of shape (1, C, H, W).
        scale (int): Scale factor (e.g., 8).
        
    Returns:
        torch.Tensor: Simulated low-resolution image.
    """
    _, _, h, w = hr.size()
    # Compute target size (ensuring integer division)
    target_h, target_w = h // scale, w // scale
    lr = F.interpolate(hr, size=(target_h, target_w), mode='bicubic', align_corners=False)
    return lr

def upsample_operator(lr, scale):
    """
    Upsample a low-resolution tensor to high-resolution using bicubic interpolation.
    
    Args:
        lr (torch.Tensor): Low-resolution image tensor of shape (1, C, H, W).
        scale (int): Scale factor (e.g., 8).
        
    Returns:
        torch.Tensor: Upsampled high-resolution image.
    """
    _, _, h, w = lr.size()
    hr = F.interpolate(lr, size=(h * scale, w * scale), mode='bicubic', align_corners=False)
    return hr

def iterative_back_projection(initial_hr, original_lr, scale, num_iter=10):
    """
    Apply iterative back-projection to refine the high-resolution image.
    
    This function simulates the low-resolution image from the current HR estimate,
    computes the error with respect to the original LR image, upscales the error,
    and corrects the HR estimate iteratively.
    
    Args:
        initial_hr (torch.Tensor): The initial high-resolution output from your SR model (1, C, H, W).
        original_lr (torch.Tensor): The original low-resolution image (1, C, H//scale, W//scale).
        scale (int): Upscaling factor.
        num_iter (int): Number of IBP iterations.
        
    Returns:
        torch.Tensor: Refined high-resolution image.
    """
    # Clone the initial HR image to avoid modifying it directly
    hr = initial_hr.clone()
    for i in range(num_iter):
        # Downsample current HR to simulate LR
        simulated_lr = degradation_operator(hr, scale)
        # Compute error between the original LR and simulated LR
        error_lr = original_lr - simulated_lr
        # Upsample the error to HR space
        error_hr = upsample_operator(error_lr, scale)
        # Update HR image by adding the upsampled error
        hr = hr + error_hr
        # Optionally, you can print the error norm to monitor convergence
        # print(f"Iteration {i+1}, error norm: {error_lr.norm().item()}")
    return hr

# Example usage:
if __name__ == '__main__':
    # Suppose you have the following tensors:
    # initial_hr: output from your SR model, shape (1, 3, H, W)
    # original_lr: the original low-res image, shape (1, 3, H//scale, W//scale)
    
    # For demonstration, we create dummy tensors:
    scale_factor = 8
    initial_hr = torch.rand(1, 3, 256, 256)       # Example HR tensor
    original_lr = torch.rand(1, 3, 256 // scale_factor, 256 // scale_factor)  # Example LR tensor
    
    # Run iterative back-projection for 10 iterations
    refined_hr = iterative_back_projection(initial_hr, original_lr, scale_factor, num_iter=10)
    
    print("Refinement done. Refined HR tensor shape:", refined_hr.shape)
