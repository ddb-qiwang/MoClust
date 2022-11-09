# assembling the whole simple/contrastive MVC model
from moclust.multiview_autoencoders import *
from .loss import *
from .fusion import *
from .ddc import *
from .optimizer import *

import torch.nn as nn

def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)

class ModelBase(nn.Module):
    def __init__(self):
        """
        Model base class
        """
        super().__init__()

        self.fusion = None
        self.optimizer = None
        self.loss = None

    def calc_losses(self, ignore_in_total=tuple()):
        return self.loss(self, ignore_in_total=ignore_in_total)

    def train_step(self, batch, epoch, it, n_batches):
        self.optimizer.zero_grad()
        _ = self(batch)
        losses = self.calc_losses()
        losses["tot"].backward()
        self.optimizer.step(epoch + it / n_batches)
        return losses

# simple MVC for single cell multiomics data
class scMVC_simple(ModelBase):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = None

        # Define Backbones and Fusion modules
        self.mvae = multiview_autoencoders(cfg.multiview_encoders_config)
        # print(self.multiview_encoders.output_sizes)
        self.fusion = get_fusion_module(cfg.fusion_config, self.mvae.output_sizes)
        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(he_init_weights)

        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, mv_input):
        self.mv_input = mv_input
        self.output_mu = self.mvae.forward(mv_input)[0]
        self.output_mean = self.mvae.forward(mv_input)[1]
        self.output_disp = self.mvae.forward(mv_input)[2]
        self.output_pi = self.mvae.forward(mv_input)[3]
        #print(self.multiview_encoders_outputs)
        self.fused = self.fusion(self.output_mu)
        self.output_soft_label, self.hidden = self.ddc(self.fused)
        self.output = [self.output_soft_label, self.output_mu, self.output_mean, self.output_disp, self.output_pi]
        return self.output

#contrastive MVC for single cell multiomics data
class scMVC_contrast(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the CoMVC model.

        :param cfg: Model config. See `config.defaults.CoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.projections = None

        # Define Backbones and Fusion modules
        self.mvae = multiview_autoencoders(cfg.multiview_encoders_config)
        # print(self.multiview_encoders.output_sizes)
        self.fusion = get_fusion_module(cfg.fusion_config, self.mvae.output_sizes)

        mvae_sizes = self.mvae.output_sizes
        assert all([mvae_sizes[0] == s for s in mvae_sizes]), f"CoMVC requires all encoders to have the same " \
                                                          f"output size. Got: {mvae_sizes}"

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = MLP(cfg.projector_config, input_size=mvae_sizes[0])

        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, mv_input):
        self.mv_input = mv_input
        self.output_mu = self.mvae.forward(mv_input)[0]
        self.output_mean = self.mvae.forward(mv_input)[1]
        self.output_disp = self.mvae.forward(mv_input)[2]
        self.output_pi = self.mvae.forward(mv_input)[3]
        #print(self.multiview_encoders_outputs)
        self.fused = self.fusion(self.output_mu)
        self.projections = self.projector(th.cat(self.output_mu, dim=0))
        self.output_soft_label, self.hidden = self.ddc(self.fused)
        self.output = [self.output_soft_label, self.output_mu, self.output_mean, self.output_disp, self.output_pi, self.fused, self.hidden]
        return self.output