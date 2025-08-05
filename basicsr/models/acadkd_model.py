import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

from .base_kd_model import BaseKD

class InfoNCELoss(nn.Module):
    """Simple InfoNCE with learned temperature."""
    def __init__(self, tau_init=0.07, mining='hard'):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.mining = mining

    def forward(self, f_s, f_t):
        """
        f_s, f_t: [B, D] normalized embeddings
        """
        B = f_s.size(0)
        sim = (f_s @ f_t.t())  # [B,B] raw dot
        sim = sim / self.tau
        # positive on diagonal
        pos = torch.diag(sim)                # [B]
        # mask out diag for negatives
        neg = sim.masked_fill(torch.eye(B, device=sim.device).bool(), -1e9)
        if self.mining == 'hard':
            # optionally pick top-k negatives per row—but here sum them all
            neg_sum = torch.logsumexp(neg, dim=1)  # [B]
        else:
            neg_sum = torch.logsumexp(neg, dim=1)
        loss = - (pos - neg_sum).mean()
        return loss
    
    
class PatchInfoNCELoss(nn.Module):
    """Patch-based InfoNCE with learned temperature."""
    def __init__(self, tau_init=0.07, patch_size=48, num_patches=8, mining='hard'):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.mining = mining
        self.patch_size = patch_size
        self.num_patches = num_patches

    def extract_patches(self, img, num_patches, patch_size):
        """
        Extract random patches from images
        Args:
            img: [B, C, H, W] 
            num_patches: number of patches per image
            patch_size: size of each patch
        Returns:
            patches: [B*num_patches, C, patch_size, patch_size]
        """
        B, C, H, W = img.shape
        patches = []
        
        for b in range(B):
            img_patches = []
            for _ in range(num_patches):
                # Random crop coordinates
                top = random.randint(0, max(0, H - patch_size))
                left = random.randint(0, max(0, W - patch_size))
                
                patch = img[b:b+1, :, top:top+patch_size, left:left+patch_size]
                img_patches.append(patch)
            
            patches.extend(img_patches)
        
        return torch.cat(patches, dim=0)  # [B*num_patches, C, patch_size, patch_size]

    def forward(self, f_s_patches, f_t_patches):
        """
        f_s_patches, f_t_patches: [B*num_patches, D] normalized patch embeddings
        """
        B_patches = f_s_patches.size(0)
        
        # Compute similarity matrix
        sim = (f_s_patches @ f_t_patches.t()) / self.tau  # [B*num_patches, B*num_patches]
        
        # Create positive mask - patches from same spatial location are positives
        pos_mask = torch.eye(B_patches, device=sim.device).bool()
        
        # Extract positive similarities
        pos_sim = sim[pos_mask]  # [B*num_patches]
        
        # Create negative mask - all other patches are negatives
        neg_mask = ~pos_mask
        neg_sim = sim.masked_fill(pos_mask, -1e9)  # mask out positives
        
        # Compute InfoNCE loss
        if self.mining == 'hard':
            # Use all negatives
            neg_sum = torch.logsumexp(neg_sim, dim=1)  # [B*num_patches]
        else:
            neg_sum = torch.logsumexp(neg_sim, dim=1)
        
        loss = -(pos_sim - neg_sum).mean()
        return loss


class HybridInfoNCELoss(nn.Module):
    """Combines image-level and patch-level contrastive learning"""
    def __init__(self, tau_init=0.07, patch_size=48, num_patches=8, 
                 image_weight=0.5, patch_weight=0.5, mining='hard'):
        super().__init__()
        self.image_loss = InfoNCELoss(tau_init, mining)
        self.patch_loss = PatchInfoNCELoss(tau_init, patch_size, num_patches, mining)
        self.image_weight = image_weight
        self.patch_weight = patch_weight
        
    def forward(self, f_s, f_t, s_pred, t_pred, proj_head):
        """
        f_s, f_t: [B, D] image-level normalized embeddings
        s_pred, t_pred: [B, C, H, W] predicted images
        proj_head: projection network for patches
        """
        # Image-level loss
        img_loss = self.image_loss(f_s, f_t)
        
        # Extract patches
        s_patches = self.patch_loss.extract_patches(s_pred, self.patch_loss.num_patches, self.patch_loss.patch_size)
        t_patches = self.patch_loss.extract_patches(t_pred, self.patch_loss.num_patches, self.patch_loss.patch_size)
        
        # Project patches
        f_s_patches = proj_head(s_patches)
        f_t_patches = proj_head(t_patches)
        
        # Handle different projection head outputs
        if len(f_s_patches.shape) > 2:
            f_s_patches = f_s_patches.flatten(1)
            f_t_patches = f_t_patches.flatten(1)
            
        # Normalize patch features
        f_s_patches = F.normalize(f_s_patches, dim=1)
        f_t_patches = F.normalize(f_t_patches, dim=1)
        
        # Patch-level loss
        patch_loss = self.patch_loss(f_s_patches, f_t_patches)
        
        # Combined loss
        total_loss = self.image_weight * img_loss + self.patch_weight * patch_loss
        
        return total_loss, img_loss, patch_loss


@MODEL_REGISTRY.register()
class ACADKD(BaseKD):
    """Adaptive Contrastive + Adversarial Distillation for SISR with Patch-based Learning."""
    def __init__(self, opt):
        super(ACADKD, self).__init__(opt)

        for p in self.net_t.parameters():
            p.requires_grad = False
            
        # --- projection head φ: Conv → GAP → FC → normalize ---
        d = opt.get('train', {}).get('contrastive_opt', {}).get('d_emb', 256)
        self.proj = nn.Sequential(
            nn.Conv2d(3, d//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d//2, d),
        ).to(self.device)
        
        # Additional projection head for patches (no global pooling)
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, d//4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, d//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d//2, d),
        ).to(self.device)

        if self.net_d is not None:
            self.net_d = self.net_d.to(self.device)

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']

        print('TRAIN OPT:', train_opt)

        # Contrastive InfoNCE (custom, not in base)
        if train_opt.get('contrastive_opt'):
            cont_opt = train_opt['contrastive_opt']
            
            # Choose contrastive learning type
            contrastive_type = cont_opt.get('type', 'hybrid')  # 'image', 'patch', or 'hybrid'
            
            if contrastive_type == 'image':
                self.cri_cont = InfoNCELoss(
                    tau_init=cont_opt.get('tau', 0.07),
                    mining=cont_opt.get('mining', 'hard')
                ).to(self.device)
            elif contrastive_type == 'patch':
                self.cri_cont = PatchInfoNCELoss(
                    tau_init=cont_opt.get('tau', 0.07),
                    patch_size=cont_opt.get('patch_size', 48),
                    num_patches=cont_opt.get('num_patches', 8),
                    mining=cont_opt.get('mining', 'hard')
                ).to(self.device)
            else:  # hybrid
                self.cri_cont = HybridInfoNCELoss(
                    tau_init=cont_opt.get('tau', 0.07),
                    patch_size=cont_opt.get('patch_size', 48),
                    num_patches=cont_opt.get('num_patches', 8),
                    image_weight=cont_opt.get('image_weight', 0.3),
                    patch_weight=cont_opt.get('patch_weight', 0.7),
                    mining=cont_opt.get('mining', 'hard')
                ).to(self.device)
                
            self.contrastive_type = contrastive_type
        else:
            self.cri_cont = None

        # GAN loss (adversarial)
        if train_opt.get('adversarial_opt'):
            self.cri_gan = build_loss(train_opt['adversarial_opt']).to(self.device)
        else:
            self.cri_gan = None

        # Discriminator optimizer
        optim_d_cfg = train_opt.get('optim_d', {'type':'Adam','lr':1e-4,'betas':[0.9,0.99]})
        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(),
            lr=optim_d_cfg['lr'], betas=optim_d_cfg.get('betas',[0.9,0.99])
        )

        if train_opt.get('resp_kd_opt'):
            self.cri_resp = build_loss(train_opt['resp_kd_opt']).to(self.device)
        else:
            self.cri_resp = None

    def optimize_parameters(self, current_iter):
        lq, gt = self.lq, self.gt

        # --- Teacher (frozen) & Student forward ---
        with torch.no_grad():
            t_pred = self.net_t(lq)
        s_pred = self.net_g(lq)

        ##### 1) Train discriminator #####
        self.net_d.train()
        self.optimizer_d.zero_grad()
        d_real = self.net_d(gt)
        loss_d_real = self.cri_gan(d_real, True, is_disc=True) if self.cri_gan else 0
        d_fake = self.net_d(s_pred.detach())
        loss_d_fake = self.cri_gan(d_fake, False, is_disc=True) if self.cri_gan else 0
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        self.optimizer_d.step()

        ##### 2) Train generator + proj head + τ #####
        self.net_d.eval()
        self.optimizer_g.zero_grad()

        loss_dict = OrderedDict()

        # (1) Reconstruction
        l_rec = self.cri_pix(s_pred, gt) if self.cri_pix else torch.tensor(0.0, device=self.device)
        loss_dict['l_rec'] = l_rec

        # (2) Response KD
        l_resp = self.cri_resp(s_pred, t_pred) if hasattr(self, 'cri_resp') and self.cri_resp else torch.tensor(0.0, device=self.device)
        loss_dict['l_resp'] = l_resp

        # (3) Contrastive Learning
        if self.cri_cont:
            if self.contrastive_type == 'image':
                # Original image-level contrastive
                f_t = self.proj(t_pred).flatten(1)
                f_s = self.proj(s_pred).flatten(1)
                f_t = F.normalize(f_t, dim=1)
                f_s = F.normalize(f_s, dim=1)
                l_cont = self.cri_cont(f_s, f_t)
                loss_dict['l_cont'] = l_cont
                
            elif self.contrastive_type == 'patch':
                # 1. Extract patches
                s_patches = self.cri_cont.extract_patches(s_pred, self.cri_cont.num_patches, self.cri_cont.patch_size)
                t_patches = self.cri_cont.extract_patches(t_pred, self.cri_cont.num_patches, self.cri_cont.patch_size)
                # 2. Project patches
                f_s_patches = self.patch_proj(s_patches)
                f_t_patches = self.patch_proj(t_patches)
                # 3. Flatten to 2D if needed
                if len(f_s_patches.shape) > 2:
                    f_s_patches = f_s_patches.flatten(1)
                    f_t_patches = f_t_patches.flatten(1)
                # 4. Normalize
                f_s_patches = F.normalize(f_s_patches, dim=1)
                f_t_patches = F.normalize(f_t_patches, dim=1)
                # 5. Compute loss
                l_cont = self.cri_cont(f_s_patches, f_t_patches)
                loss_dict['l_cont'] = l_cont
                
            else:  # hybrid
                # Both image and patch level
                f_t = self.proj(t_pred).flatten(1)
                f_s = self.proj(s_pred).flatten(1)
                f_t = F.normalize(f_t, dim=1)
                f_s = F.normalize(f_s, dim=1)
                
                l_cont, l_img, l_patch = self.cri_cont(f_s, f_t, s_pred, t_pred, self.patch_proj)
                loss_dict['l_cont'] = l_cont
                loss_dict['l_cont_img'] = l_img
                loss_dict['l_cont_patch'] = l_patch
        else:
            l_cont = torch.tensor(0.0, device=self.device)
            loss_dict['l_cont'] = l_cont

        # (4) Adversarial (fool D)
        d_fake_for_g = self.net_d(s_pred)
        l_adv = self.cri_gan(d_fake_for_g, True, is_disc=False) if self.cri_gan else torch.tensor(0.0, device=self.device)
        loss_dict['l_adv'] = l_adv

        # total (use weights from config)
        w = self.opt['train']
        loss_G = (
            w.get('pixel_opt', {}).get('loss_weight', 1.0) * l_rec
          + w.get('resp_kd_opt', {}).get('loss_weight', 0.1) * l_resp
          + w.get('contrastive_opt', {}).get('loss_weight', 0.01) * l_cont
          + w.get('adversarial_opt', {}).get('loss_weight', 0.001) * l_adv
        )
        loss_G.backward()
        self.optimizer_g.step()

        # log & EMA
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
            