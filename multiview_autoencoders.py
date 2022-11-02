# comstruct network
# an Encoder for each view of data
import torch.nn as nn
import torch.nn.functional as F
import torch as th

# ZINB loss
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = th.lgamma(disp+eps) + th.lgamma(x+1.0) - th.lgamma(x+disp+eps)
        t2 = (disp+x) * th.log(1.0 + (mean/(disp+eps))) + (x * (th.log(disp+eps) - th.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - th.log(1.0-pi+eps)
        zero_nb = th.pow(disp/(disp+mean+eps), disp)
        zero_case = -th.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = th.where(th.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*th.square(pi)
            result += ridge
        
        result = th.mean(result)
        return result

    
# basic layer functions
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * th.randn_like(x)
        return x

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return th.clamp(th.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return th.clamp(F.softplus(x), min=1e-4, max=1e4)

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class Backbone(nn.Module):
    def __init__(self):
        """
        Backbone base class
        """
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# define multi layer perceptron class for projection
class MLP(Backbone):
    def __init__(self, cfg, input_size=None, **_):
        """
        MLP backbone

        :param cfg: MLP config
        :type cfg: config.defaults.MLP
        :param input_size: Optional input size which overrides the one set in `cfg`.
        :type input_size: Optional[Union[List, Tuple]]
        :param _:
        :type _:
        """
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size)

    @staticmethod
    def get_activation_module(a):
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"Invalid MLP activation: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = list(input_size)
        else:
            output_size = list(cfg.input_size)

        if len(output_size) > 1:
            layer_container.append(nn.Flatten())
            output_size = [np.prod(output_size)]

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for n_units, act, _use_bias, _use_bn in zip(cfg.layers, activations, use_bias, use_bn):
            # If we get n_units = -1, then the number of units should be the same as the previous number of units, or
            # the input dim.
            if n_units == -1:
                n_units = output_size[0]

            layer_container.append(nn.Linear(in_features=output_size[0], out_features=n_units, bias=_use_bias))
            if _use_bn:
                # Add BN before activation
                layer_container.append(nn.BatchNorm1d(num_features=n_units))
            if act is not None:
                # Add activation
                layer_container.append(cls.get_activation_module(act))
            output_size[0] = n_units

        return output_size
    

# encoder function
class Autoencoder(nn.Module):
    
    def __init__(self,cfg):
        """cfg: Layer = [Input_dim, ..., Latent_dim]
                activation = "ReLU"
        """ 
        super(Autoencoder, self).__init__()
        self.Layer = cfg.Layer
        self.activation = cfg.activation
        self.encoder = buildNetwork(self.Layer[:-1], type="encode", activation=self.activation)
        self.decoder = buildNetwork(self.Layer[:0:-1], type="decode", activation=self.activation)
        self._enc_mu = nn.Linear(self.Layer[-2], self.Layer[-1])
        self._dec_mean = nn.Sequential(nn.Linear(self.Layer[1], self.Layer[0]), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.Layer[1], self.Layer[0]), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.Layer[1], self.Layer[0]), nn.Sigmoid())
        self.zinb_loss = ZINBLoss().cuda()
    
    def output_size(self):
        return self.Layer[-1]
      
    def forward(self, x):
        h = self.encoder(x+th.randn_like(x) * 1.5)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return [z0, _mean, _disp, _pi]
 

# multiview autoencoder function
class multiview_autoencoders(nn.Module):
    
    def __init__(self, mvae_configs):

        super().__init__()

        self.mvae = nn.ModuleList()
        for cfg in mvae_configs: 
            self.mvae.append(Autoencoder(cfg[1]))

    @property
    def output_sizes(self):
        return [ec.output_size() for ec in self.mvae]


    def forward(self, mv_input):
        views = mv_input[0]
        assert len(views) == len(self.mvae)
        outputs_latent = [ae(v)[0] for ae, v in zip(self.mvae, views)]
        outputs_zinb_mean = [ae(v)[1] for ae, v in zip(self.mvae, views)]
        outputs_zinb_disp = [ae(v)[2] for ae, v in zip(self.mvae, views)]
        outputs_zinb_pi = [ae(v)[3] for ae, v in zip(self.mvae, views)]
        return [outputs_latent, outputs_zinb_mean, outputs_zinb_disp, outputs_zinb_pi]