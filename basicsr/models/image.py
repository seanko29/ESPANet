import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

# from neosr.archs import build_network
from basicsr.archs import build_network

# from neosr.models.base import base
from basicsr.models.base_model import BaseModel

# from neosr.utils.registry import MODEL_REGISTRY
from basicsr.utils.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    
@MODEL_REGISTRY.register()
# class image(base)
class image(BaseModel):
    """Single-Image Super-Resolution model."""

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__(opt)

        # define network net_g
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)  # type: ignore[reportArgumentType,reportArgumentType,arg-type]
        if self.opt["path"].get("print_network", False) is True:
            self.print_network(self.net_g)

        # define network net_d
        self.net_d = self.opt.get("network_d", None)
        if self.net_d is not None:
            self.net_d = build_network(self.opt["network_d"])
            self.net_d = self.model_to_device(self.net_d)  # type: ignore[reportArgumentType]
            if self.opt.get("print_network", False) is True:
                self.print_network(self.net_d)
                
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self) -> None:
        # enable ECO optimization:
        self.eco = self.opt["train"].get("eco", False)
        # ECO alpha scheduling
        self.eco_schedule = self.opt["train"].get("eco_schedule", "sigmoid")
        # ECO amount of iters
        self.eco_iters = self.opt["train"].get("eco_iters", 80000)
        # ECO init iters
        self.eco_init = self.opt["train"].get("eco_init", 15000)
    
    
    def eco_strategy(self, current_iter: int):
        """Adapted version of "Empirical Centroid-oriented Optimization":
        https://arxiv.org/abs/2312.17526.
        """
        with torch.no_grad():
            # define alpha with sigmoid-like curve, slope/skew at 0.25
            if self.eco_schedule == "sigmoid":
                a = 1 / (
                    1 + math.exp(-1 * (10 * (current_iter / self.eco_iters - 0.25)))
                )
            else:
                a = min(current_iter / self.eco_iters, 1.0)
            # network prediction
            self.net_output = self.net_g(self.lq)  # type: ignore[reportCallIssue,operator]
            # define gt centroid
            self.gt = ((1 - a) * self.net_output) + (a * self.gt)
            # downsampled prediction
            self.lq_scaled = torch.clamp(
                F.interpolate(
                    self.net_output,
                    scale_factor=1 / self.scale,
                    mode="bicubic",
                    antialias=True,
                ),
                0,
                1,
            )
            # define lq centroid
            self.output = ((1 - a) * self.lq_scaled) + (a * self.lq)
        # predict from lq centroid
        self.output = self.net_g(self.output)  # type: ignore[reportCallIssue,operator]

        return self.output, self.gt

    def closure(self, current_iter: int):
        if self.net_d is not None:
            for p in self.net_d.parameters():  # type: ignore[reportAttributeAccessIssue,operator]
                p.requires_grad = False

        # increment accumulation counter
        self.n_accumulated += 1
        # reset accumulation counter
        if self.n_accumulated >= self.accum_iters:
            self.n_accumulated = 0

        with torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # eco
            if self.eco and current_iter <= self.eco_iters:
                if current_iter < self.eco_init and self.pretrain is None:
                    self.output = self.net_g(self.lq)  # type: ignore[reportCallIssue,operator]
                else:
                    self.output, self.gt = self.eco_strategy(current_iter)
                    self.gt = torch.clamp(self.gt, 1 / 255, 1)
            else:
                self.output = self.net_g(self.lq)  # type: ignore[reportCallIssue,operator]
            self.output = torch.clamp(self.output, 1 / 255, 1)



