
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

# Third-party libraries for DWT
try:
    from pytorch_wavelets import DWTForward
except ImportError:
    # Placeholder if pytorch_wavelets is not installed
    class DWTForward:
        def __init__(self, J, wave):
            print("Warning: pytorch_wavelets not found. Using placeholder DWT.")
        def __call__(self, x):
            b, c, h, w = x.shape
            return torch.zeros(b, c, h//2, w//2).to(x.device), \
                   [torch.zeros(b, c, 3, h//2, w//2).to(x.device)]

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel

import math

# --- PFA-KD Specific Modules and Losses ---

def unwrap(net):
    return net.module if isinstance(net, DistributedDataParallel) else net

class CustomizationProjector(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(CustomizationProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )
    def forward(self, x):
        return self.projector(x)

class DWTLoss(nn.Module):
    def __init__(self, wave='haar', loss_weight=1.0):
        super(DWTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.dwt = DWTForward(J=1, wave=wave)
    def forward(self, pred_feat, target_feat):
        _, pred_hf_list = self.dwt(pred_feat)
        _, target_hf_list = self.dwt(target_feat)
        return F.l1_loss(pred_hf_list[0], target_hf_list[0]) * self.loss_weight

# --- BaseKD Model for Inheritance ---

class BaseKD(SRModel):
    def __init__(self, opt):
        # The teacher network is now created in the child class to solve initialization order issues.
        super(BaseKD, self).__init__(opt)

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None
        if self.cri_pix is None and self.cri_kd is None:
            raise ValueError('All pixel, KD, and perceptual losses are None.')


# --- Main PFA-KD Model ---

@MODEL_REGISTRY.register()
class PAFKDNewModel(BaseKD):
    def __init__(self, opt):
        super(PAFKDNewModel, self).__init__(opt)
        self.persistent_log_dict = OrderedDict()

    def init_training_settings(self):
        # =========================================================================
        # START OF ATTRIBUTEERROR FIX
        # =========================================================================
        # 1. Create the Teacher Network here, BEFORE the parent's init_training_settings is called.
        self.net_t = build_network(self.opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)
        load_path_t = self.opt['path'].get('pretrain_network_t', None)
        assert load_path_t is not None, "Checkpoint of the teacher network is required."
        param_key = self.opt['path'].get('param_key_t', 'params')
        self.load_network(self.net_t, load_path_t, self.opt['path'].get('strict_load_t', True), param_key)
        self.net_t.eval()
        
        # 2. Now call the parent's setup, which initializes optimizers, losses for the student (net_g).
        super().init_training_settings()
        # =========================================================================
        # END OF ATTRIBUTEERROR FIX
        # =========================================================================

        train_opt = self.opt['train']
        pfa_opt = train_opt['pfa_opt']

        # Dynamically infer channel numbers from the models to avoid YAML mismatch.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
            # unwrap DDP if needed
            if isinstance(self.net_t, torch.nn.parallel.DistributedDataParallel):
                teacher_net = self.net_t.module
            else:
                teacher_net = self.net_t
            if isinstance(self.net_g, torch.nn.parallel.DistributedDataParallel):
                student_net = self.net_g.module
            else:
                student_net = self.net_g

            teacher_ch = teacher_net.get_features(dummy_input).shape[1]
            student_ch = student_net.get_features(dummy_input).shape[1]
            
        print("\n" + "="*60)
        print(f"INFO: Dynamically inferred feature channels for PFA-HETERO-KD:")
        print(f"  - Teacher Feature Channels (inferred): {teacher_ch}")
        print(f"  - Student Feature Channels (inferred): {student_ch}")
        print("="*60 + "\n")

        self.projector = CustomizationProjector(
            in_channels=teacher_ch,
            out_channels=student_ch
        ).to(self.device)

        self.temp_reconstructor = nn.Conv2d(
            student_ch, 3, 3, 1, 1
        ).to(self.device)

        if self.opt['train'].get('gan_opt'):
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.cri_gan = build_loss(self.opt['train']['gan_opt']).to(self.device)

        self.cri_custom = build_loss(pfa_opt['custom_opt']).to(self.device)


        self.optimizer_proj = torch.optim.Adam(
            list(self.projector.parameters()) + list(self.temp_reconstructor.parameters()),
            lr=train_opt['optim_proj']['lr']
        )
        self.optimizers.append(self.optimizer_proj)

        if self.opt['train'].get('gan_opt'):
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(), lr=train_opt['optim_d']['lr']
            )
            self.optimizers.append(self.optimizer_d)
            
    def compute_dynamic_weight(self, current_iter, total_iter, w_init, w_final, p_mid=0.5, k=10):
        progress = current_iter / total_iter
        weight = w_final + (w_init - w_final) / (1 + math.exp(k * (progress - p_mid)))
        return weight

    def optimize_parameters(self, current_iter):
        if current_iter % 2 == 0:
            self.run_alignment_stage(current_iter)
        else:
            self.run_distillation_stage(current_iter)
        self.log_dict = self.reduce_loss_dict(self.persistent_log_dict)

    def run_alignment_stage(self, current_iter):
        if self.opt['rank'] == 0:
            print(f"\r[Iter {current_iter}] ==> STAGE 1: Running Perceptual Alignment (Training Projector)...", end='')
        
        # Set projector to training mode, student to eval mode
        self.projector.train()
        self.temp_reconstructor.train()
        self.net_g.eval()  # Freeze student during projector training
        
        self.optimizer_proj.zero_grad()
        
        with torch.no_grad():
            # Extract teacher features (f_t)
            feat_t = unwrap(self.net_t).get_features(self.lq)
            # Get student output for comparison
            output_s = self.net_g(self.lq)
        
        # Project teacher features through customization head (θ_t^h) to get f̃_t
        feat_t_custom = self.projector(feat_t)
        
        # Feed customized features through temp reconstructor (simulating student head)
        output_temp = self.temp_reconstructor(feat_t_custom)
        output_temp_upsampled = F.interpolate(output_temp, size=output_s.shape[2:], mode='bicubic', align_corners=False)
        
        # Compute loss: align customized teacher features with student output
        # This is the key part of CustomKD's projector training
        l_align = self.cri_custom(output_temp_upsampled, output_s.detach())
        
        # Add pixel loss to ensure the projected features can reconstruct the image
        l_pix_align = self.cri_pix(output_temp_upsampled, self.gt)
        
        # Total alignment loss
        l_total_align = l_align + l_pix_align
        
        # Backward pass to train the projector
        l_total_align.backward()
        self.optimizer_proj.step()
        
        # Log alignment losses
        self.persistent_log_dict['l_align'] = l_align
        self.persistent_log_dict['l_pix_align'] = l_pix_align

    def run_distillation_stage(self, current_iter):
        if self.opt['rank'] == 0:
            print(f"\r[Iter {current_iter}] ==> STAGE 2: Running Student Distillation...", end='')
        
        # Set projector to eval mode, student to training mode
        self.projector.eval()
        self.net_g.train()
        
        if self.opt['train'].get('gan_opt'):
            for p in self.net_d.parameters():
                p.requires_grad = False
        
        self.optimizer_g.zero_grad()

        # Forward pass: Teacher path (following CustomKD)
        with torch.no_grad():
            # Extract teacher features (f_t)
            feat_t = unwrap(self.net_t).get_features(self.lq)
            # Project through trained customization head to get f̃_t
            feat_t_aligned = self.projector(feat_t)

        # Forward pass: Student path (following CustomKD)
        # Extract student features (f_s)
        feat_s = unwrap(self.net_g).get_features(self.lq)
        # Get student output
        self.output = self.net_g(self.lq)

        total_iter = self.opt['train']['total_iter']

        if self.opt['train'].get('use_dynamic_weights', False):
            # Dynamic weighting
            w_pix = self.opt['train']['pixel_opt']['loss_weight']
            w_custom = self.compute_dynamic_weight(current_iter, total_iter, w_init=2.0, w_final=0.5)
            w_gan = self.compute_dynamic_weight(current_iter, total_iter, w_init=0.005, w_final=0.02)
        else:
            # Constant weighting
            w_pix = self.opt['train']['pixel_opt']['loss_weight']
            w_custom = self.opt['train']['pfa_opt']['custom_opt']['loss_weight']
            if self.opt['train'].get('gan_opt'):
                w_gan = self.opt['train']['gan_opt']['loss_weight']
            else:
                w_gan = 0

        l_g_total = 0
        loss_dict = OrderedDict()

        # Pixel loss (task-specific supervision)
        l_pix = self.cri_pix(self.output, self.gt) * w_pix
        l_g_total += l_pix
        loss_dict['l_pix'] = l_pix

        # Customized feature distillation (following CustomKD Eq.2)
        # Distill from customized teacher features (f̃_t) to student features (f_s)
        l_custom = self.cri_custom(feat_s, feat_t_aligned) * w_custom
        l_g_total += l_custom
        loss_dict['l_custom'] = l_custom

        # GAN loss (if enabled)
        if self.opt['train'].get('gan_opt'):
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, target_is_real=True, is_disc=False) * w_gan
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()
            
        # Discriminator training (if GAN is enabled)
        if self.opt['train'].get('gan_opt'):
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, target_is_real=True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, target_is_real=False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            l_d_total = l_d_real + l_d_fake
            l_d_total.backward()
            self.optimizer_d.step()
            
        self.persistent_log_dict.update(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
