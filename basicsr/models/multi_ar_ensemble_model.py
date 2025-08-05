import torch
from torch import nn
import torch.nn.functional as F
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY, ARCH_REGISTRY
import copy

@MODEL_REGISTRY.register()
class MultiArchEnsembleSRModel(SRModel):
    """
    A multi-architecture ensemble SR model.
    
    It loads different sub-models (e.g., HAT, SwinIR, HMA, NAFNet, etc.) from separate checkpoints,
    applies test-time augmentation (self-ensemble) for each model if enabled, and then fuses their outputs.
    
    The configuration should include a list under network_g.sub_models. For each sub-model, you must specify:
      - arch: The name of the registered architecture (e.g., "HATNet", "SwinIR", "HMANet", etc.)
      - checkpoint: Path to the pretrained weights.
      - args: (Optional) Dictionary of arguments to pass to the model constructor.
      - weight: (Optional) Relative weight (default 1.0).
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.sub_models = nn.ModuleList()
        self.sub_weights = []
        
        sub_models_cfg = opt['network_g'].get('sub_models', [])
        if len(sub_models_cfg) == 0:
            raise ValueError("No sub_models defined in the config under network_g.sub_models")
        
        for sub_cfg in sub_models_cfg:
            arch_type = sub_cfg['arch']
            checkpoint_path = sub_cfg['checkpoint']
            args = sub_cfg.get('args', {})
            weight = sub_cfg.get('weight', 1.0)
            self.sub_weights.append(weight)
            
            model_cls = ARCH_REGISTRY.get(arch_type)
            if model_cls is None:
                raise ValueError(f"Architecture {arch_type} is not registered in ARCH_REGISTRY")
            sub_model = model_cls(**args)
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'params_ema' in checkpoint:
                sub_model.load_state_dict(checkpoint['params_ema'], strict=True)
            else:
                sub_model.load_state_dict(checkpoint['params'], strict=True)
            sub_model.eval()  # set model to eval mode
            self.sub_models.append(sub_model)
        
        # Normalize weights
        total_weight = sum(self.sub_weights)
        self.sub_weights = [w / total_weight for w in self.sub_weights]
    
    def forward(self, lq):
        """
        lq: low-resolution input tensor.
        Returns the fused super-resolved output.
        """
        sr_outputs = []
        use_self_ensemble = self.opt['val'].get("self_ensemble", False)
        for sub_model in self.sub_models:
            with torch.no_grad():
                if use_self_ensemble and hasattr(sub_model, 'forward'):
                    sr = self.run_self_ensemble(sub_model, lq)
                else:
                    sr = sub_model(lq)
            sr_outputs.append(sr)
        
        # Weighted fusion of outputs (assume all outputs have same shape)
        ensemble_sr = 0
        for weight, sr in zip(self.sub_weights, sr_outputs):
            ensemble_sr += weight * sr
        return ensemble_sr

    def run_self_ensemble(self, model, lq):
        """
        Perform self-ensemble (eight-fold augmentation) for a given model.
        This follows the typical flip/transpose strategy.
        """
        def _transform(t, op):
            if op == 'v':  # flip horizontally (last dimension)
                return torch.flip(t, dims=[3])
            elif op == 'h':  # flip vertically (height dimension)
                return torch.flip(t, dims=[2])
            elif op == 't':  # transpose height and width
                return t.transpose(2, 3)
            else:
                return t

        # Create augmented versions: start with original
        img_list = [lq]
        # For each op in the set, augment all images in the current list.
        for op in ['v', 'h', 't']:
            aug_list = [_transform(t, op) for t in img_list]
            img_list.extend(aug_list)
        
        # Now run inference on each augmented image
        out_list = []
        for aug in img_list:
            with torch.no_grad():
                out = model(aug)
            out_list.append(out)
        
        # Invert the transformations on the outputs.
        # The following logic mirrors the pattern used in many SR self-ensemble implementations.
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        
        # Average all outputs
        ensemble_out = torch.mean(torch.stack(out_list, dim=0), dim=0)
        return ensemble_out
