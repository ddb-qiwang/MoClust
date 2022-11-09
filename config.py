import torch
from .preprocess import *
from typing import Tuple, List, Union, Optional
from typing_extensions import Literal
from pydantic import BaseModel
import torch as th
import numpy as np

# configs
class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__
    
class encoder_cfg(Config):
    # Layer dimensions
    Layer: Tuple[Union[int, str], ...] = (2000,256,64,32)
    # Activation function
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"

class mvencoder_cfg(Config):
    # atac encoder cfg
    view1_encoder_cfg = encoder_cfg(Layer=(20000,256,64,32))
    # rna encoder cfg
    view2_encoder_cfg = encoder_cfg(Layer=(14,32))
    
class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str
    # Number of samples to load. Set to None to load all samples
    n_samples: int = None
    # Subset of views to load. Set to None to load all views
    select_views: Tuple[int, ...] = None
    # Subset of labels (classes) to load. Set to None to load all classes
    select_labels: Tuple[int, ...] = None
    # Number of samples to load for each class. Set to None to load all samples
    label_counts: Tuple[int, ...] = None
    # Standard deviation of noise added to the views `noise_views`.
    noise_sd: float = None
    # Subset of views to add noise to
    noise_views: Tuple[int, ...] = None


class Loss_config(Config):
    # Number of views
    n_views: int = 2
    # Number of clusters
    n_clusters: int = None
    # cuda device
    device = 0
    # Terms to use in the loss, separated by '|'. E.g. "ddc_1|ddc_2|ddc_3|" for the DDC clustering loss
    funcs: str
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Multiplication factor for the sigma hyperparameter
    rel_sigma = 0.15
    # Tau hyperparameter
    tau = 0.1
    # Delta hyperparameter
    delta = 0.1
    # if zinb_loss
    if_zinb = True
    # Gamma hyperparametre
    gamma = 2.0
    # index for views modeled by ZINB
    zinb_index = list([0])
    # Fraction of batch size to use as the number of negative samples in the contrastive loss. Set to -1 to use all
    # pairs (except the positive) as negative pairs.
    negative_samples_ratio: float = 0.25
    # Similarity function for the contrastive loss. "cos" (default) and "gauss" are supported.
    contrastive_similarity: Literal["cos", "gauss"] = "cos"
    # Enable the adaptive contrastive weighting?
    adaptive_contrastive_weight = True


class Optimizer_config(Config):
    # Base learning rate
    learning_rate: float = 0.0001
    # Max gradient norm for gradient clipping.
    clip_norm: float = 5.0
    # Step size for the learning rate scheduler. None disables the scheduler.
    scheduler_step_size: int = None
    # Multiplication factor for the learning rate scheduler
    scheduler_gamma: float = 0.1


class DDC_config(Config):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden = 32
    # Use batch norm after the first fully connected layer?
    use_bn = False
    # If direct or not
    direct = False
    # cuda device
    device = 0

    
class Fusion_config(Config):
    # Fusion method. "mean" constant weights = 1/V. "weighted_mean": Weighted average with learned weights.
    method: Literal["mean", "weighted_mean"]
    # Number of views
    n_views: int

    
class scMVC_simple_config(Config):
    # multiview_encodoers_config
    multiview_encoders_config: mvencoder_cfg = mvencoder_cfg()
    # fusion config
    fusion_config: Fusion_config = Fusion_config(method="weighted_mean", n_views=2)
    # Clustering module config.
    cm_config: Union[DDC_config] = DDC_config(n_clusters=7)
    # Loss function config
    loss_config: Loss_config = Loss_config(n_clusters=7, funcs="ddc_1|ddc_2|ddc_3",)
    # Optimizer config
    optimizer_config = Optimizer_config()

class MLP_config(Config):
    # Shape of the input
    input_size: Tuple[int, ...] = None
    # Units in the network layers
    layers: Tuple[Union[int, str], ...] = (512, 512, 256)
    # Activation function. Can be a single string specifying the activation function for all layers, or a list/tuple of
    # string specifying the activation function for each layer.
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"
    # Include bias parameters? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bias: Union[bool, Tuple[bool, ...]] = True
    # Include batch norm after layers? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bn: Union[bool, Tuple[bool, ...]] = False    

class scMVC_contrast_config(Config):
    # multiview_encodoers_config
    multiview_encoders_config: mvencoder_cfg = mvencoder_cfg()
    # Projection head config. Set to None to remove the projection head.
    projector_config: Optional[MLP_config] = None
    # Fusion module config.
    fusion_config:  Fusion_config = Fusion_config(method="weighted_mean", n_views=2)
    # Clustering module config.
    cm_config: Union[DDC_config] = DDC_config(n_clusters=7)
    # Loss function config
    loss_config:  Loss_config = Loss_config(n_clusters=7, funcs="ddc_1|ddc_2|ddc_3|contrast",)
    # Optimizer config
    optimizer_config = Optimizer_config()