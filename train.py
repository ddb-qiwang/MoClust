# training
from .helpers import *
from .evaluate import *
from torch.autograd import Variable

n_epochs = 200

def train(net, loader, n_epochs):

    n_batches = len(loader)
    for e in range(1, n_epochs + 1):
        iter_losses = []
        for i, data in enumerate(loader):
            batch = data
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
                #print(batch_losses)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(npy(batch_losses))

def train_cfg(net, loader, n_epochs, eval_data, batch_size, eval_interval):

    n_batches = len(loader)
    for e in range(1, n_epochs + 1):
        iter_losses = []
        for i, data in enumerate(loader):
            batch, _ = data
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(npy(batch_losses))
        logs = get_logs(net, eval_data=eval_data, batch_size=batch_size, eval_interval=eval_interval,
                        iter_losses=iter_losses, epoch=e, include_params=True)
        print(logs)