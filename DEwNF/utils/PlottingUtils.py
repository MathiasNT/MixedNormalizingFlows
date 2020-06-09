import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

def plot_4_contexts_cond_flow(flow_dist, contexts, scaler, n_samples=256):
    assert contexts.shape[0] == 4, 'Need 4 contexts inorder to create 4 plots'

    fig, axs = plt.subplots(2, 2)
    for i in range(4):
        cur_axs = axs[i // 2, i % 2]
        cond_dist = flow_dist.condition(contexts[i])
        x_s = cond_dist.sample((n_samples,)).cpu()
        if scaler is not None:
            x_s = scaler.inverse_transform(x_s)
        cur_axs.scatter(x_s[:, 0], x_s[:, 1], c='b', s=5)
        cur_axs.set_xlim(-6, 6)
        cur_axs.set_ylim(-6, 6)
        context_str = np.array2string(contexts[i].cpu().numpy(), precision=2, separator=',')
        cur_axs.set_title(f"Context: {context_str}")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def plot_loss(train_loss_arr, test_loss_arr):
    """
    Makes nice plots of the train and test loss
    :param train_loss_arr: array
    :param test_loss_arr: array
    :return:
    """
    plt.figure()
    plt.title("Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_arr)
    plt.plot(test_loss_arr, '--')
    plt.legend(['Train', 'Test'])
    plt.show()


def sliding_plot_loss(train_loss_arr, test_loss_arr, window_size):
    """
    Makes nice plots of the train and test loss
    :param train_loss_arr: array
    :param test_loss_arr: array
    :return:
    """
    plt.figure()
    plt.title("Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_arr[-window_size:])
    plt.plot(test_loss_arr[-window_size:], '--')
    plt.legend(['Train', 'Test'])
    plt.show()


def plot_train_results(train_loss_arr, test_loss_arr, no_noise_arr, start_idx=0, end_idx=None, freq = 10):
    """
    Makes nice plots of the train and test loss
    :param train_loss_arr: array
    :param test_loss_arr: array
    :param no_noise_arr: array
    :param start_idx: int
    :param end_idx: int
    :param freq: int
    :return:
    """
    if end_idx is None:
        end_idx = len(train_loss_arr)

    ticks = [x*10 for x in range(len(train_loss_arr))]

    plt.figure()
    plt.title("Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(ticks, train_loss_arr[start_idx:end_idx])
    plt.plot(ticks, test_loss_arr[start_idx:end_idx], '--')
    plt.plot(ticks, no_noise_arr[start_idx:end_idx], '--')
    plt.legend(['Train', 'Test', 'No noise train'])
    plt.show()


def plot_samples(plot_flow_dist, x_plot, scaler, n_samples=256):
    x_s = plot_flow_dist.sample((n_samples,))
    if x_s.is_cuda:
        x_s = x_s.cpu()
    if scaler is not None:
        x_s = scaler.inverse_transform(x_s)
    plt.figure(figsize=(4, 4))
    plt.scatter(x_plot[:, 0], x_plot[:, 1], c='r', s=5)
    plt.scatter(x_s[:, 0], x_s[:, 1], c='b', s=5)
    plt.axis('equal')
    plt.show()


def create_overlay(shape, bounds, flow_dist):
    cm = matplotlib.cm.get_cmap('cividis')
    nlats, nlons = shape

    lats_array = torch.linspace(start=bounds[1][0], end=bounds[0][0], steps=nlats)
    lons_array = torch.linspace(start=bounds[0][1], end=bounds[1][1], steps=nlons)
    x, y = torch.meshgrid(lats_array, lons_array)

    points = torch.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    data = flow_dist.log_prob(points).reshape(nlats, nlons).cpu().detach().numpy()

    data = np.exp(data)
    data = (data - data.min()) / (data.max() - data.min())

    overlay = cm(data)
    return lons_array, lats_array, overlay
