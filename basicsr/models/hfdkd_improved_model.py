# HFDKD_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from einops import rearrange

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from .base_kd_model import BaseKD


def unwrap(model):
    """Unwrap DDP model if needed"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


class TokenToFeatureMap(nn.Module):
    """Convert ViT tokens to CNN feature maps with spatial awareness"""
    def __init__(self, target_size=None):
        super().__init__()
        self.target_size = target_size
        
    def forward(self, tokens, target_size=None):
        if target_size is None:
            target_size = self.target_size
            
        # Handle different token formats
        if len(tokens.shape) == 3:  # (B, N, C) - standard ViT tokens
            B, N, C = tokens.shape
            H = W = int(np.sqrt(N))
            # Reshape to spatial format
            x = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)
        else:  # Already in spatial format (B, C, H, W)
            x = tokens
        
        # Interpolate to target size if needed
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class FrequencyProjector(nn.Module):
    """Project features in frequency domain for better high-frequency recovery"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim * 2, out_dim, 1, 1, 0)  # *2 for real and imaginary
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.token_converter = TokenToFeatureMap()
        
    def forward(self, tokens, target_size):
        # Convert to feature map
        x = self.token_converter(tokens, target_size)
        
        # FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag
        
        # Concatenate real and imaginary parts
        x_freq = torch.cat([x_fft_real, x_fft_imag], dim=1)
        
        # Project in frequency domain
        x_freq = self.conv1(x_freq)
        x_freq = self.conv2(x_freq)
        
        # IFFT back to spatial domain
        x_spatial = torch.fft.ifft2(x_freq, dim=(-2, -1)).real
        
        return x_spatial


class HeterogeneousFeatureAligner(nn.Module):
    """Align features between ViT and CNN for SR with multi-scale processing"""
    def __init__(self, vit_dim, cnn_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # Multi-scale projectors for different resolution levels
        self.scale_projectors = nn.ModuleList()
        for i in range(num_scales):
            self.scale_projectors.append(
                nn.Sequential(
                    nn.Conv2d(vit_dim, cnn_dim, 3, 1, 1),
                    nn.BatchNorm2d(cnn_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cnn_dim, cnn_dim, 3, 1, 1),
                    nn.BatchNorm2d(cnn_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Frequency-aware projection
        self.freq_projector = FrequencyProjector(vit_dim, cnn_dim)
        
        # Token converter
        self.token_converter = TokenToFeatureMap()
        
        # Attention-based fusion
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(cnn_dim * (num_scales + 1), cnn_dim, 1, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, vit_features, target_size):
        aligned_features = []
        
        # Multi-scale alignment
        for i, proj in enumerate(self.scale_projectors):
            scale = 2 ** i
            if scale == 1:
                scaled_size = target_size
            else:
                scaled_size = (target_size[0] // scale, target_size[1] // scale)
            
            # Convert tokens to feature map
            feat_map = self.token_converter(vit_features, scaled_size)
            
            # Project to CNN space
            aligned = proj(feat_map)
            
            # Upsample to target size if needed
            if i > 0:
                aligned = F.interpolate(aligned, size=target_size, mode='bilinear', align_corners=False)
            
            aligned_features.append(aligned)
        
        # Add frequency-aware features
        freq_features = self.freq_projector(vit_features, target_size)
        aligned_features.append(freq_features)
        
        # Attention-based fusion
        all_features = torch.cat(aligned_features, dim=1)
        attention_weights = self.attention_fusion(all_features)
        
        # Weighted combination
        final_features = sum(feat * attention_weights for feat in aligned_features)
        
        return final_features


class SRReconstructionHead(nn.Module):
    """Reconstruction head for SR tasks"""
    def __init__(self, in_channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Progressive upsampling
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * (scale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(in_channels, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        return self.upsampler(x)


class HFDSRLoss(nn.Module):
    """Comprehensive loss function for HFD-SR"""
    def __init__(self, lambda_pixel=1.0, lambda_feat=10.0, lambda_custom=10.0, lambda_freq=1.0, lambda_feat_freq=1.0):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_feat = lambda_feat
        self.lambda_custom = lambda_custom
        self.lambda_freq = lambda_freq
        self.lambda_feat_freq = lambda_feat_freq
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, student_output, teacher_output, student_features, teacher_features, 
                customized_features, target_hr=None):
        
        loss = 0
        loss_dict = {}
        
        # Pixel-level loss
        loss_pixel = self.l1_loss(student_output, teacher_output)
        loss += self.lambda_pixel * loss_pixel
        loss_dict['l_pixel'] = loss_pixel
        
        # Feature distillation loss (task-general)
        if student_features.get('projected') is not None:
            # Ensure spatial dimensions match between teacher and student features
            student_proj = student_features['projected']
            teacher_feat = teacher_features
            
            # Align spatial dimensions if they don't match
            if student_proj.shape[-2:] != teacher_feat.shape[-2:]:
                # Resize teacher features to match student features
                teacher_feat = F.interpolate(
                    teacher_feat, 
                    size=student_proj.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            loss_feat = self.l2_loss(student_proj, teacher_feat)
            loss += self.lambda_feat * loss_feat
            loss_dict['l_feat'] = loss_feat
        
        # Customized feature loss (task-specific)
        if customized_features is not None:
            # Ensure spatial dimensions match for customized features
            student_orig = student_features['original']
            custom_feat = customized_features
            
            if student_orig.shape[-2:] != custom_feat.shape[-2:]:
                # Resize customized features to match student features
                custom_feat = F.interpolate(
                    custom_feat, 
                    size=student_orig.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            loss_custom = self.l2_loss(student_orig, custom_feat)
            loss += self.lambda_custom * loss_custom
            loss_dict['l_custom'] = loss_custom
            
            # NEW: Customized feature loss (task-specific, FREQUENCY domain)
            loss_feat_freq = self.frequency_loss(student_orig, custom_feat)
            loss += self.lambda_feat_freq * loss_feat_freq
            loss_dict['l_feat_freq'] = loss_feat_freq

        # Frequency loss for high-frequency recovery
        loss_freq = self.frequency_loss(student_output, teacher_output)
        loss += self.lambda_freq * loss_freq
        loss_dict['l_freq'] = loss_freq
        
        # Ground truth loss if available
        if target_hr is not None:
            loss_gt = self.l1_loss(student_output, target_hr)
            loss += loss_gt
            loss_dict['l_gt'] = loss_gt
        
        return loss, loss_dict
    
    def frequency_loss(self, student_output, teacher_output):
        """Compute loss in frequency domain for better high-frequency recovery"""
        # Ensure spatial dimensions match
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            # Resize teacher output to match student output
            teacher_output = F.interpolate(
                teacher_output, 
                size=student_output.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # FFT
        student_fft = torch.fft.fft2(student_output, dim=(-2, -1))
        teacher_fft = torch.fft.fft2(teacher_output, dim=(-2, -1))
        
        # High-frequency mask (emphasize high frequencies)
        B, C, H, W = student_output.shape
        freq_mask = self.create_frequency_mask(H, W).to(student_output.device)
        
        # Weighted frequency loss
        loss = torch.mean(freq_mask * torch.abs(student_fft - teacher_fft))
        
        return loss
    
    def create_frequency_mask(self, H, W):
        """Create a mask that emphasizes high frequencies"""
        h_half, w_half = H // 2, W // 2
        
        # Create distance matrix from center
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y = y.float() - h_half
        x = x.float() - w_half
        dist = torch.sqrt(x**2 + y**2)
        
        # Normalize and create mask (higher weight for higher frequencies)
        max_dist = torch.sqrt(torch.tensor(h_half**2 + w_half**2))
        mask = dist / max_dist
        mask = torch.clamp(mask, 0.1, 1.0)  # Avoid zero weights
        
        return mask.unsqueeze(0).unsqueeze(0)


@MODEL_REGISTRY.register()
class HFDKDImproved(BaseKD):
    def __init__(self, opt):
        # ✅ Create teacher network BEFORE calling parent
        self._pre_init_setup(opt)
        
        # Now call parent initialization
        super(HFDKDImproved, self).__init__(opt)
        self.persistent_log_dict = OrderedDict()
        self.current_stage = "distillation"
        self.l1_loss = nn.L1Loss()
        
    def _pre_init_setup(self, opt):
        """Setup required before parent initialization"""
        self.opt = opt
        # Set device first
        self.device = torch.device('cuda' if opt.get('num_gpu') else 'cpu')
        
        # Create teacher network before parent init
        self._create_teacher_network()
        
    def _create_teacher_network(self):
        """Create and load teacher network properly"""
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        
        # Check if teacher network config exists
        if 'network_t' not in self.opt:
            raise ValueError(
                "Teacher network configuration 'network_t' not found in config. "
                "Please add 'network_t' section to your YAML configuration file."
            )
        
        logger.info("Creating teacher network...")
        
        # Build teacher network
        self.net_t = build_network(self.opt['network_t'])
        
        # Move to device (simple version since full model_to_device isn't available yet)
        if hasattr(self, 'device'):
            self.net_t = self.net_t.to(self.device)
        else:
            # Fallback device detection
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net_t = self.net_t.to(device)
            
        logger.info(f"Teacher network created: {self.net_t.__class__.__name__}")
        
        # Load pretrained teacher weights
        load_path = self.opt['path'].get('pretrain_network_t')
        if load_path is not None:
            logger.info(f"Loading teacher network from: {load_path}")
            param_key = self.opt['path'].get('param_key_t', 'params')
            strict_load = self.opt['path'].get('strict_load_t', True)
            
            try:
                # Simple loading without load_network method (not available yet)
                checkpoint = torch.load(load_path, map_location='cpu')
                
                # Extract parameters
                if param_key in checkpoint:
                    state_dict = checkpoint[param_key]
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Use custom loading method if available (for EDSR)
                teacher_net = unwrap(self.net_t)
                if hasattr(teacher_net, 'load_pretrained_weights'):
                    logger.info("Using custom weight loading for teacher network")
                    missing_keys, unexpected_keys = teacher_net.load_pretrained_weights(state_dict)
                    
                    if missing_keys:
                        logger.warning(f"Missing keys when loading teacher weights: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading teacher weights: {unexpected_keys}")
                else:
                    # Standard loading for other architectures
                    if strict_load:
                        self.net_t.load_state_dict(state_dict)
                    else:
                        self.net_t.load_state_dict(state_dict, strict=False)
                    
                logger.info("Teacher network loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to load teacher network: {str(e)}")
                logger.error("Please check:")
                logger.error(f"  - File exists: {load_path}")
                logger.error(f"  - Parameter key: {param_key}")
                logger.error(f"  - Network architecture matches pretrained weights")
                raise
        else:
            logger.warning("No pretrained teacher network path specified.")
        
        # Set teacher to eval mode and freeze parameters
        self.net_t.eval()
        for param in self.net_t.parameters():
            param.requires_grad = False
        
        logger.info("Teacher network setup completed!")

    def init_training_settings(self):
        # Call parent's init first
        super().init_training_settings()
        
        # Handle student weight loading for EDSRStudent architecture
        self._load_student_weights()
        
        train_opt = self.opt['train']
        hfd_opt = train_opt.get('hfd_opt', {})
        
        # ✅ Now this check should pass since we created net_t early
        if not hasattr(self, 'net_t') or self.net_t is None:
            raise RuntimeError(
                "Teacher network 'net_t' not properly initialized. "
                "This should not happen after _create_teacher_network(). "
                "Please check your configuration."
            )
        
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        logger.info("Initializing HFDKD training settings...")
        
        # Dynamically infer channel numbers from the models
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
                
                # Get unwrapped models
                teacher_net = unwrap(self.net_t)
                student_net = unwrap(self.net_g)
                
                # Extract feature dimensions with better error handling
                try:
                    if hasattr(teacher_net, 'get_features'):
                        teacher_features = teacher_net.get_features(dummy_input)
                        if len(teacher_features.shape) == 3:  # ViT tokens (B, N, C)
                            teacher_dim = teacher_features.shape[-1]
                        else:  # CNN features (B, C, H, W)
                            teacher_dim = teacher_features.shape[1]
                    else:
                        # Try to infer from model architecture
                        if hasattr(teacher_net, 'embed_dim'):
                            teacher_dim = teacher_net.embed_dim
                        elif hasattr(teacher_net, 'num_features'):
                            teacher_dim = teacher_net.num_features
                        else:
                            teacher_dim = hfd_opt.get('teacher_dim', 180)  # Default for HAT/SwinIR
                        
                    if hasattr(student_net, 'get_features'):
                        student_features = student_net.get_features(dummy_input)
                        student_dim = student_features.shape[1]
                    else:
                        # Try to infer from model architecture
                        if hasattr(student_net, 'num_feat'):
                            student_dim = student_net.num_feat
                        elif hasattr(student_net, 'num_features'):
                            student_dim = student_net.num_features
                        else:
                            student_dim = hfd_opt.get('student_dim', 64)  # Default for EDSR
                        
                except Exception as e:
                    logger.warning(f"Could not infer dimensions automatically: {e}")
                    teacher_dim = hfd_opt.get('teacher_dim', 180)  # Default for HAT/SwinIR
                    student_dim = hfd_opt.get('student_dim', 64)   # Default for EDSR
                    
        except Exception as e:
            logger.error(f"Error during feature dimension inference: {e}")
            teacher_dim = hfd_opt.get('teacher_dim', 180)
            student_dim = hfd_opt.get('student_dim', 64)
        
        logger.info(f"Feature dimensions - Teacher: {teacher_dim}, Student: {student_dim}")
        
        # Debug: Print model architecture info
        logger.info(f"Teacher model type: {type(teacher_net).__name__}")
        logger.info(f"Student model type: {type(student_net).__name__}")
        
        # Check if models have expected attributes
        if hasattr(teacher_net, 'embed_dim'):
            logger.info(f"Teacher embed_dim: {teacher_net.embed_dim}")
        if hasattr(student_net, 'num_feat'):
            logger.info(f"Student num_feat: {student_net.num_feat}")
        
        # Initialize HFD components
        self.feature_aligner = HeterogeneousFeatureAligner(
            vit_dim=teacher_dim,
            cnn_dim=student_dim,
            num_scales=hfd_opt.get('num_scales', 3)
        ).to(self.device)
        
        # Student feature projector for task-general distillation
        self.student_projector = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, 1, 1, 0),
            nn.BatchNorm2d(teacher_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dim, teacher_dim, 3, 1, 1)
        ).to(self.device)
        
        # Reconstruction head for alignment stage
        self.temp_reconstructor = SRReconstructionHead(
            in_channels=student_dim,
            scale_factor=self.opt.get('scale', 4)
        ).to(self.device)
        
        # HFD-specific loss
        self.cri_hfd = HFDSRLoss(
            lambda_pixel=hfd_opt.get('lambda_pixel', 1.0),
            lambda_feat=hfd_opt.get('lambda_feat', 10.0),
            lambda_custom=hfd_opt.get('lambda_custom', 10.0),
            lambda_freq=hfd_opt.get('lambda_freq', 0.1)
        ).to(self.device)
        
        # Additional optimizer for HFD components
        hfd_params = (
            list(self.feature_aligner.parameters()) +
            list(self.student_projector.parameters()) +
            list(self.temp_reconstructor.parameters())
        )
        optim_hfd_opt = train_opt.get('optim_hfd', {})
        self.optimizer_hfd = torch.optim.Adam(
            hfd_params,
            lr=optim_hfd_opt.get('lr', 1e-4),
            betas=optim_hfd_opt.get('betas', [0.9, 0.999])
        )
        self.optimizers.append(self.optimizer_hfd)
        
        # Training configuration
        self.alignment_frequency = hfd_opt.get('alignment_frequency', 5)
        self.use_dynamic_weights = hfd_opt.get('use_dynamic_weights', True)
        
        logger.info("HFDKD initialization completed successfully!")

    def _load_student_weights(self):
        """Load pretrained weights for student model, handling EDSRStudent architecture"""
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        
        # Check if student pretrained weights path exists
        load_path = self.opt['path'].get('pretrain_network_g')
        if load_path is None:
            logger.info("No pretrained student weights specified. Starting from scratch.")
            return
        
        logger.info(f"Loading student network from: {load_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(load_path, map_location='cpu')
            
            # Extract parameters
            param_key = self.opt['path'].get('param_key_g', 'params')
            if param_key in checkpoint:
                state_dict = checkpoint[param_key]
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Check if student is EDSRStudent
            student_net = unwrap(self.net_g)
            if hasattr(student_net, 'load_pretrained_weights'):
                # Use the custom loading method for EDSRStudent
                logger.info("Using custom weight loading for EDSRStudent architecture")
                missing_keys, unexpected_keys = student_net.load_pretrained_weights(state_dict)
                
                if missing_keys:
                    logger.warning(f"Missing keys when loading student weights: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading student weights: {unexpected_keys}")
            else:
                # Use standard loading for other architectures
                strict_load = self.opt['path'].get('strict_load_g', True)
                if strict_load:
                    self.net_g.load_state_dict(state_dict)
                else:
                    self.net_g.load_state_dict(state_dict, strict=False)
            
            logger.info("Student network loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load student network: {str(e)}")
            logger.error("Please check:")
            logger.error(f"  - File exists: {load_path}")
            logger.error(f"  - Parameter key: {param_key}")
            logger.error(f"  - Network architecture matches pretrained weights")
            raise

    def extract_features(self, model, x, is_teacher=False):
        """Extract features from model"""
        # Check if teacher network exists when extracting teacher features
        if is_teacher and not hasattr(self, 'net_t'):
            raise AttributeError(
                "Teacher network 'net_t' not found. Please ensure that 'network_t' is defined in your configuration."
            )
        
        model_unwrapped = unwrap(model)
        
        if hasattr(model_unwrapped, 'get_features'):
            features = model_unwrapped.get_features(x)
        else:
            # Fallback: extract from intermediate layers
            features = None
            def hook_fn(module, input, output):
                nonlocal features
                features = output
            
            # Register hook on a middle layer (model-specific)
            if is_teacher:
                # For ViT-based models (HAT, SwinIR), extract from transformer blocks
                target_layer = None
                for name, module in model_unwrapped.named_modules():
                    if 'layers' in name or 'blocks' in name or 'transformer' in name:
                        target_layer = module
                        break
                
                # If no transformer layer found, try to find any layer with reasonable output size
                if target_layer is None:
                    for name, module in model_unwrapped.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)) and 'embed' in name:
                            target_layer = module
                            break
                
                # For HAT specifically, try to find the right layer
                if target_layer is None:
                    for name, module in model_unwrapped.named_modules():
                        if 'layers' in name and len(list(module.children())) > 0:
                            # Take the first layer of the transformer
                            target_layer = list(module.children())[0]
                            break
            else:
                # For CNN-based models, extract from feature extraction part
                target_layer = None
                for name, module in model_unwrapped.named_modules():
                    if 'body' in name or 'features' in name:
                        target_layer = module
                        break
            
            if target_layer is not None:
                hook = target_layer.register_forward_hook(hook_fn)
                _ = model(x)
                hook.remove()
            
            if features is None:
                # Ultimate fallback: use consistent dimensions based on model type
                if is_teacher:
                    # For teacher (HAT/SwinIR), use the configured teacher_dim
                    teacher_dim = self.opt['train'].get('hfd_opt', {}).get('teacher_dim', 180)
                    features = torch.randn(x.shape[0], teacher_dim, x.shape[2]//4, x.shape[3]//4).to(x.device)
                else:
                    # For student (EDSR), use the configured student_dim
                    student_dim = self.opt['train'].get('hfd_opt', {}).get('student_dim', 64)
                    features = torch.randn(x.shape[0], student_dim, x.shape[2]//4, x.shape[3]//4).to(x.device)
        
        # Ensure features have the expected format and dimensions
        if len(features.shape) == 3:  # (B, N, C) - ViT tokens
            B, N, C = features.shape
            H = W = int(np.sqrt(N))
            if H * W == N:  # Valid square shape
                features = rearrange(features, 'b (h w) c -> b c h w', h=H, w=W)
            else:
                # If not a perfect square, reshape to a reasonable spatial size
                H = W = int(np.sqrt(N)) if int(np.sqrt(N))**2 == N else int(np.sqrt(N)) + 1
                features = features[:, :H*W, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Debug: Log feature dimensions
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if is_teacher:
            logger.debug(f"Teacher features shape: {features.shape}")
        else:
            logger.debug(f"Student features shape: {features.shape}")
        
        return features
    
    def compute_dynamic_weight(self, current_iter, total_iter, w_init, w_final, p_mid=0.5, k=10):
        """Compute dynamic weight based on training progress"""
        progress = current_iter / total_iter
        weight = w_final + (w_init - w_final) / (1 + math.exp(k * (progress - p_mid)))
        return weight


     # this is the customKD implementation version
    def run_alignment_stage(self, current_iter):
        """
        STAGE 1 (IMPROVED): Feature Customization guided by the student's reconstruction head.
        """
        if self.opt.get('rank', 0) == 0:
            print(f"\r[Iter {current_iter}] ==> STAGE 1 (CustomKD): Student-Guided Alignment...", end='')

        # Set modes: Train aligner, freeze student
        self.feature_aligner.train()
        self.net_g.eval() # Keep the entire student model frozen

        self.optimizer_hfd.zero_grad()

        # --- Step 1: Get teacher features (no change) ---
        with torch.no_grad():
            teacher_features = self.extract_features(self.net_t, self.lq, is_teacher=True)
            # We need the student's feature shape for target_size
            student_features_shape = self.extract_features(self.net_g, self.lq, is_teacher=False).shape

        # --- Step 2: Align teacher features (no change) ---
        target_size = student_features_shape[-2:]
        aligned_teacher_features = self.feature_aligner(teacher_features, target_size)

        # --- Step 3: Guide alignment using the student's frozen reconstruction head ---
        # Unwrap the student model to access its parts
        student_model_unwrapped = unwrap(self.net_g)
        
        # Check if student has reconstruction head (new unified EDSR)
        if hasattr(student_model_unwrapped, 'get_reconstruction_head'):
            reconstruction_head = student_model_unwrapped.get_reconstruction_head()
        elif hasattr(student_model_unwrapped, 'reconstruction_head'):
            reconstruction_head = student_model_unwrapped.reconstruction_head
        else:
            raise AttributeError(
                "Student model must have a reconstruction_head for CustomKD. "
                "Use EDSR with use_reconstruction_head=True for the student network."
            )

        # Pass the ALIGNED teacher features through the STUDENT's frozen head
        # This is the core of the CustomKD philosophy
        reconstructed_output = reconstruction_head(aligned_teacher_features)

        # --- Step 4: Calculate loss against the ground truth ---
        # The loss function now directly optimizes the final image quality
        l_total_align = self.l1_loss(reconstructed_output, self.gt)

        l_total_align.backward()
        self.optimizer_hfd.step()

        # Log the alignment loss
        self.persistent_log_dict['l_align_customkd'] = l_total_align
    
    def run_distillation_stage(self, current_iter):
        """Stage 2: Knowledge Distillation - Train student with aligned features"""
        if self.opt.get('rank', 0) == 0:
            print(f"\r[Iter {current_iter}] ==> STAGE 2: Knowledge Distillation...", end='')
        
        # Check if teacher network exists
        if not hasattr(self, 'net_t'):
            raise AttributeError(
                "Teacher network 'net_t' not found. Please ensure that 'network_t' is defined in your configuration."
            )
        
        # Set modes
        self.feature_aligner.eval()
        self.net_g.train()
        
        self.optimizer_g.zero_grad()
        
        # Forward pass - Teacher
        with torch.no_grad():
            self.output_t = self.net_t(self.lq)
            teacher_features = self.extract_features(self.net_t, self.lq, is_teacher=True)
            
            # Get aligned teacher features - use student features size for proper alignment
            # We'll set this after extracting student features to ensure proper alignment
            aligned_teacher_features = None  # Will be set after student features extraction
        
        # Forward pass - Student
        self.output = self.net_g(self.lq)
        student_features = self.extract_features(self.net_g, self.lq, is_teacher=False)
        
        # Now align teacher features to match student features spatial size
        if aligned_teacher_features is None:
            target_size = student_features.shape[-2:] if len(student_features.shape) == 4 else (64, 64)
            aligned_teacher_features = self.feature_aligner(teacher_features, target_size)
        
        # Project student features for general distillation
        student_features_projected = self.student_projector(student_features)
        
        # Debug: Log feature dimensions
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        logger.debug(f"Teacher features shape: {teacher_features.shape}")
        logger.debug(f"Student features shape: {student_features.shape}")
        logger.debug(f"Student projected features shape: {student_features_projected.shape}")
        logger.debug(f"Aligned teacher features shape: {aligned_teacher_features.shape}")
        logger.debug(f"Target size for alignment: {target_size}")
        
        # Dynamic weighting
        if self.use_dynamic_weights:
            total_iter = self.opt['train']['total_iter']
            w_custom = self.compute_dynamic_weight(current_iter, total_iter, w_init=2.0, w_final=0.5)
        else:
            w_custom = self.opt['train'].get('hfd_opt', {}).get('lambda_custom', 10.0)
        
        l_g_total = 0
        loss_dict = OrderedDict()
        
        # Standard losses from BaseKD
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_kd:
            l_kd = self.cri_kd(self.output, self.output_t)
            l_g_total += l_kd
            loss_dict['l_kd'] = l_kd
        
        # HFD-specific losses
        features_dict = {
            'original': student_features,
            'projected': student_features_projected
        }
        
        hfd_loss, hfd_loss_dict = self.cri_hfd(
            self.output, self.output_t,
            features_dict, teacher_features,
            aligned_teacher_features, self.gt
        )
        
        # Weight HFD loss
        l_g_total += w_custom * hfd_loss
        
        # Add HFD losses to log
        for k, v in hfd_loss_dict.items():
            loss_dict[f'hfd_{k}'] = v
        
        # Perceptual loss if available
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        l_g_total.backward()
        self.optimizer_g.step()
        
        self.persistent_log_dict.update(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def optimize_parameters(self, current_iter):
        """Main optimization function with alternating stages"""
        # Determine current stage
        if current_iter % self.alignment_frequency == 0:
            self.current_stage = "alignment"
            self.run_alignment_stage(current_iter)
        else:
            self.current_stage = "distillation"
            self.run_distillation_stage(current_iter)
        
        # Update log dict
        self.log_dict = self.reduce_loss_dict(self.persistent_log_dict)
    
    def get_current_log(self):
        """Override to include stage information"""
        log_dict = super().get_current_log()
        log_dict['stage'] = self.current_stage
        return log_dict