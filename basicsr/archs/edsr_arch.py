import torch
from torch import nn as nn

from basicsr.archs.arch_utils import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """
    EDSR network that can work as both teacher and student.
    
    - As TEACHER: Uses original architecture (upscale + conv_last)
    - As STUDENT: Can optionally use reconstruction_head for CustomKD training
    - After training: Always uses original architecture for inference
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 use_reconstruction_head=False):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.use_reconstruction_head = use_reconstruction_head

        # --- Feature Extractor Components (same for both) ---
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualBlockNoBN,
            num_block,
            num_feat=num_feat,
            res_scale=res_scale,
            pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # --- Reconstruction Components ---
        if use_reconstruction_head:
            # For student training: grouped reconstruction head
            self.reconstruction_head = nn.Sequential(
                Upsample(upscale, num_feat),
                nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            )
            # Keep original layers for weight loading compatibility
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            # For teacher/inference: original architecture
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def get_features(self, x):
        """
        Extracts deep features from the main body of the network.
        This is the output of the "feature extractor" part.
        """
        # Normalize the input image
        self.mean = self.mean.type_as(x)
        x_norm = (x - self.mean) * self.img_range

        # Pass through the feature extraction part
        feat_shallow = self.conv_first(x_norm)
        feat_deep = self.conv_after_body(self.body(feat_shallow))
        
        # The final features are the sum from the residual connection
        features = feat_deep + feat_shallow
        return features

    def forward(self, x):
        """
        Forward pass that works for both teacher and student.
        """
        # Normalize the input image
        self.mean = self.mean.type_as(x)
        x_norm = (x - self.mean) * self.img_range

        # Feature extraction
        feat_shallow = self.conv_first(x_norm)
        feat_deep = self.conv_after_body(self.body(feat_shallow))
        feat = feat_deep + feat_shallow

        # Reconstruction
        if self.use_reconstruction_head:
            # Use reconstruction head (for student training)
            out = self.reconstruction_head(feat)
        else:
            # Use original architecture (for teacher/inference)
            out = self.conv_last(self.upsample(feat))

        # Denormalize the output
        out = out / self.img_range + self.mean

        return out
    
    def get_reconstruction_head(self):
        """
        Get the reconstruction head for CustomKD training.
        This allows external access to the reconstruction head during training.
        """
        if not self.use_reconstruction_head:
            raise RuntimeError("Reconstruction head not available. Set use_reconstruction_head=True during initialization.")
        return self.reconstruction_head
    
    def load_pretrained_weights(self, pretrained_state_dict):
        """
        Load pretrained weights, handling both architectures and naming mismatches.
        """
        # Create a mapping for reconstruction head if needed
        if self.use_reconstruction_head:
            state_dict = {}
            
            for key, value in pretrained_state_dict.items():
                if key.startswith('conv_first') or key.startswith('body') or key.startswith('conv_after_body'):
                    # These layers exist in both architectures
                    state_dict[key] = value
                elif key.startswith('upsample'):
                    # Handle upsample -> upscale mapping (pretrained weights use 'upsample')
                    new_key = key.replace('upsample', 'upscale')
                    state_dict[new_key] = value
                    # Also map to reconstruction_head
                    recon_key = key.replace('upsample', 'reconstruction_head.0')
                    state_dict[recon_key] = value
                elif key.startswith('upscale'):
                    # Handle upscale -> reconstruction_head mapping
                    new_key = key.replace('upscale', 'reconstruction_head.0')
                    state_dict[new_key] = value
                    # Also keep original for compatibility
                    state_dict[key] = value
                elif key.startswith('conv_last'):
                    # Map conv_last to reconstruction_head.1
                    new_key = key.replace('conv_last', 'reconstruction_head.1')
                    state_dict[new_key] = value
                    # Also keep original for compatibility
                    state_dict[key] = value
                else:
                    # Skip other keys that don't exist in new architecture
                    continue
            
            # Load the mapped weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        else:
            # Standard loading for teacher/inference, but handle upsample -> upscale mapping
            state_dict = {}
            for key, value in pretrained_state_dict.items():
                if key.startswith('upsample'):
                    # Handle upsample -> upscale mapping
                    new_key = key.replace('upsample', 'upscale')
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value
            
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


# Keep the old EDSRStudent for backward compatibility, but mark as deprecated
@ARCH_REGISTRY.register()
class EDSRStudent(EDSR):
    """
    @deprecated: Use EDSR with use_reconstruction_head=True instead.
    This class is kept for backward compatibility.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "EDSRStudent is deprecated. Use EDSR with use_reconstruction_head=True instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, use_reconstruction_head=True, **kwargs)