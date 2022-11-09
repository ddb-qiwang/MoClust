import os
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from .helpers import *

IGNORE_IN_TOTAL = ("contrast",)


def calc_metrics(labels, pred):
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat = ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
        "ari": adjusted_rand_score(labels, pred),
    }
    return metrics


def get_log_params(net):
    """
    Get the network parameters we want to log.

    :param net: Model
    :type net:
    :return:
    :rtype:
    """
    params_dict = {}
    weights = []
    if getattr(net, "fusion", None) is not None:
        with th.no_grad():
            weights = net.fusion.get_weights(softmax=True)

    elif hasattr(net, "attention"):
        weights = net.weights

    for i, w in enumerate(npy(weights)):
        params_dict[f"fusion/weight_{i}"] = w

    if hasattr(net, "discriminators"):
        for i, discriminator in enumerate(net.discriminators):
            d0, dv = npy([discriminator.d0, discriminator.dv])
            params_dict[f"discriminator_{i}/d0/mean"] = d0.mean()
            params_dict[f"discriminator_{i}/d0/std"] = d0.std()
            params_dict[f"discriminator_{i}/dv/mean"] = dv.mean()
            params_dict[f"discriminator_{i}/dv/std"] = dv.std()

    return params_dict


def get_eval_data(dataset, n_eval_samples, batch_size):
    """
    Create a dataloader to use for evaluation

    :param dataset: Inout dataset.
    :type dataset: th.utils.data.Dataset
    :param n_eval_samples: Number of samples to include in the evaluation dataset. Set to None to use all available
                           samples.
    :type n_eval_samples: int
    :param batch_size: Batch size used for training.
    :type batch_size: int
    :return: Evaluation dataset loader
    :rtype: th.utils.data.DataLoader
    """
    if n_eval_samples is not None:
        *views, labels = dataset.tensors
        n = views[0].size(0)
        idx = np.random.choice(n, min(n, n_eval_samples), replace=False)
        views, labels = [v[idx] for v in views], labels[idx]
        dataset = th.utils.data.TensorDataset(*views, labels)

    eval_loader = th.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=0,
                                           drop_last=False, pin_memory=False)
    return eval_loader


def batch_predict(net, eval_data, batch_size, if_train=True, if_latent=False, if_recon=False, if_softlabel=False):
    """
    Compute predictions for `eval_data` in batches. Batching does not influence predictions, but it influences the loss
    computations.

    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloader
    :type eval_data: th.utils.data.DataLoader
    :param batch_size: Batch size
    :type batch_size: int
    :return: Label tensor, predictions tensor, list of dicts with loss values, array containing mean and std of cluster
             sizes.
    :rtype:
    """
    input_x = []
    input_y = []
    predictions = []
    softlabel = []
    labels = []
    losses = []
    cluster_sizes = []
    latent_features = []
    fused_features = []
    hidden_features = []
    totalmeanx = []
    totalmeany = []
    totaldispx = []
    totaldispy = []
    totalpix = []
    totalpiy = []

    net.eval()
    with th.no_grad():
        for i, (batch, label) in enumerate(eval_data):
            pred = net(batch)[0]
            latent = net(batch)[1]
            mean = net(batch)[2]
            disp = net(batch)[3]
            pi = net(batch)[4]
            fused = net(batch)[5]
            hidden = net(batch)[6]
            input_x.append(npy(batch[0][0]))
            input_y.append(npy(batch[0][1]))
            labels.append(npy(label))
            softlabel.append(npy(pred))
            predictions.append(npy(pred).argmax(axis=1))
            latent_features.append(npy(latent))
            fused_features.append(npy(fused))
            hidden_features.append(npy(hidden))
            totalmeanx.append(npy(mean[0]))
            totalmeany.append(npy(mean[1]))
            totaldispx.append(npy(disp[0]))
            totaldispy.append(npy(disp[1]))
            totalpix.append(npy(pi[0]))
            totalpiy.append(npy(pi[1]))

            # Only calculate losses for full batches
            if label.size(0) == batch_size:
                batch_losses = net.calc_losses(ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(npy(batch_losses))
                cluster_sizes.append(npy(pred.sum(dim=0)))

    input_x = np.concatenate(input_x, axis=0)
    input_y = np.concatenate(input_y, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    softlabel = np.concatenate(softlabel, axis=0)
    latent_features = np.concatenate(latent_features, axis=1)
    fused_features = np.concatenate(fused_features, axis=0)
    hidden_features = np.concatenate(hidden_features, axis=0)
    totalmeanx = np.concatenate(totalmeanx, axis=0)
    totaldispx = np.concatenate(totaldispx, axis=0)
    totalpix = np.concatenate(totalpix, axis=0)
    totalmeany = np.concatenate(totalmeany, axis=0)
    totaldispy = np.concatenate(totaldispy, axis=0)
    totalpiy = np.concatenate(totalpiy, axis=0)
    if if_recon:
        return input_x, input_y, totalmeanx, totalmeany, totaldispx, totaldispy, totalpix, totalpiy
    if if_latent:
        return labels, predictions, latent_features, fused_features, hidden_features
    if if_softlabel:
        return softlabel
    if if_train:
        net.train()
    return labels, predictions, losses, np.array(cluster_sizes).sum(axis=0)

def batch_predict_nolabel(net, eval_data, if_train=True, if_latent=False, if_recon=False, if_softlabel=False):
    """
    Compute predictions for `eval_data` in batches. Batching does not influence predictions, but it influences the loss
    computations.

    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloader
    :type eval_data: th.utils.data.DataLoader
    :param batch_size: Batch size
    :type batch_size: int
    :return: Label tensor, predictions tensor, list of dicts with loss values, array containing mean and std of cluster
             sizes.
    :rtype:
    """
    input_x = []
    input_y = []
    predictions = []
    softlabel = []
    losses = []
    cluster_sizes = []
    latent_features = []
    fused_features = []
    hidden_features = []
    totalmeanx = []
    totalmeany = []
    totaldispx = []
    totaldispy = []
    totalpix = []
    totalpiy = []

    net.eval()
    with th.no_grad():
        for i, batch in enumerate(eval_data):
            pred = net(batch)[0]
            latent = net(batch)[1]
            mean = net(batch)[2]
            disp = net(batch)[3]
            pi = net(batch)[4]
            fused = net(batch)[5]
            hidden = net(batch)[6]
            input_x.append(npy(batch[0][0]))
            input_y.append(npy(batch[0][1]))
            softlabel.append(npy(pred))
            predictions.append(npy(pred).argmax(axis=1))
            latent_features.append(npy(latent))
            fused_features.append(npy(fused))
            hidden_features.append(npy(hidden))
            totalmeanx.append(npy(mean[0]))
            totalmeany.append(npy(mean[1]))
            totaldispx.append(npy(disp[0]))
            totaldispy.append(npy(disp[1]))
            totalpix.append(npy(pi[0]))
            totalpiy.append(npy(pi[1]))

    if if_recon:
        return input_x, input_y, totalmeanx, totalmeany, totaldispx, totaldispy, totalpix, totalpiy
    if if_latent:
        return predictions, latent_features, fused_features, hidden_features
    if if_softlabel:
        return softlabel
    if if_train:
        net.train()
    return predictions, losses, np.array(cluster_sizes).sum(axis=0)

def get_logs(net, eval_data, batch_size, eval_interval, iter_losses=None, epoch=None, include_params=True):
    if iter_losses is not None:
        logs = add_prefix(dict_means(iter_losses), "iter_losses")
    else:
        logs = {}
    if (epoch is None) or ((epoch % eval_interval) == 0):
        labels, pred, eval_losses, cluster_sizes = batch_predict(net, eval_data, batch_size)
        eval_losses = dict_means(eval_losses)
        logs.update(add_prefix(eval_losses, "eval_losses"))
        logs.update(add_prefix(calc_metrics(labels, pred), "metrics"))
        logs.update(add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
    if include_params:
        logs.update(add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch
    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True):
    """
    Evaluate a training run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param cfg_name: Config name
    :type cfg_name: str
    :param experiment_identifier: 8-character unique identifier for the current experiment
    :type experiment_identifier: str
    :param run: Run to evaluate
    :type run: int
    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloder
    :type eval_data: th.utils.data.DataLoader
    :param callbacks: List of callbacks to call after evaluation
    :type callbacks: List
    :param load_best: Load the "best.pt" model before evaluation?
    :type load_best: bool
    :return: Evaluation logs
    :rtype: dict
    """
    if load_best:
        model_path = get_save_dir(cfg_name, experiment_identifier, run) / "best.pt"
        if os.path.isfile(model_path):
            net.load_state_dict(th.load(model_path))
        else:
            print(f"Unable to load best model for evaluation. Model file not found: {model_path}")
    logs = get_logs(cfg, net, eval_data, include_params=True)
    for cb in callbacks:
        cb.at_eval(net=net, logs=logs)
    return logs


def plot_projection(X, method, hue, ax, title=None, cmap="tab10", legend_title=None, legend_loc=1, **kwargs):
    X = project(X, method)
    pl = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=hue, ax=ax, legend="full", palette=cmap, **kwargs)
    leg = pl.get_legend()
    leg._loc = legend_loc
    if title is not None:
        ax.set_title(title)
    if legend_title is not None:
        leg.set_title(legend_title)


def project(X, method):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2).fit_transform(X)
    elif method is None:
        return X
    else:
        raise RuntimeError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="cfg_name", required=True)
    parser.add_argument("-t", "--tag", dest="tag", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_experiment(args.cfg_name, args.tag, args.plot)
