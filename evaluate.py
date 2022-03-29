import os
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import helpers

IGNORE_IN_TOTAL = ("contrast",)


def calc_metrics(labels, pred):

    acc, cmat = helpers.ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
        "ari": adjusted_rand_score(labels, pred),
    }
    return metrics


def get_log_params(net):

    params_dict = {}
    weights = []
    if getattr(net, "fusion", None) is not None:
        with th.no_grad():
            weights = net.fusion.get_weights(softmax=True)

    elif hasattr(net, "attention"):
        weights = net.weights

    for i, w in enumerate(helpers.npy(weights)):
        params_dict[f"fusion/weight_{i}"] = w

    if hasattr(net, "discriminators"):
        for i, discriminator in enumerate(net.discriminators):
            d0, dv = helpers.npy([discriminator.d0, discriminator.dv])
            params_dict[f"discriminator_{i}/d0/mean"] = d0.mean()
            params_dict[f"discriminator_{i}/d0/std"] = d0.std()
            params_dict[f"discriminator_{i}/dv/mean"] = dv.mean()
            params_dict[f"discriminator_{i}/dv/std"] = dv.std()

    return params_dict


def get_eval_data(dataset, n_eval_samples, batch_size):

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
            input_x.append(helpers.npy(batch[0][0]))
            input_y.append(helpers.npy(batch[0][1]))
            labels.append(helpers.npy(label))
            softlabel.append(helpers.npy(pred))
            predictions.append(helpers.npy(pred).argmax(axis=1))
            latent_features.append(helpers.npy(latent))
            fused_features.append(helpers.npy(fused))
            hidden_features.append(helpers.npy(hidden))
            totalmeanx.append(helpers.npy(mean[0]))
            totalmeany.append(helpers.npy(mean[1]))
            totaldispx.append(helpers.npy(disp[0]))
            totaldispy.append(helpers.npy(disp[1]))
            totalpix.append(helpers.npy(pi[0]))
            totalpiy.append(helpers.npy(pi[1]))

            # Only calculate losses for full batches
            if label.size(0) == batch_size:
                batch_losses = net.calc_losses(ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(helpers.npy(batch_losses))
                cluster_sizes.append(helpers.npy(pred.sum(dim=0)))

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


def get_logs(net, eval_data, batch_size, eval_interval, iter_losses=None, epoch=None, include_params=True):
    if iter_losses is not None:
        logs = helpers.add_prefix(helpers.dict_means(iter_losses), "iter_losses")
    else:
        logs = {}
    if (epoch is None) or ((epoch % eval_interval) == 0):
        labels, pred, eval_losses, cluster_sizes = batch_predict(net, eval_data, batch_size)
        eval_losses = helpers.dict_means(eval_losses)
        logs.update(helpers.add_prefix(eval_losses, "eval_losses"))
        logs.update(helpers.add_prefix(calc_metrics(labels, pred), "metrics"))
        logs.update(helpers.add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
    if include_params:
        logs.update(helpers.add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch
    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True):

    if load_best:
        model_path = helpers.get_save_dir(cfg_name, experiment_identifier, run) / "best.pt"
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
