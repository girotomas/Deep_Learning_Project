from matplotlib.colors import ListedColormap, colorConverter
from torch import FloatTensor, LongTensor, ByteTensor
import matplotlib.pyplot as plt
import numpy as np


# Compute numbers of mispredicted labels
def compute_nb_errors(model, t_input, target):
    output = model.forward(t_input)
    if output.shape[1] > 1:  # target class in one hot encoding
        target = target.type(LongTensor)
        m_o, i_o = output.max(1)
        m_t, i_t = target.max(1)
        return sum(i_o.ne(i_t))
    else:  # used for target with only one dimension between 0 and #classes
        target = target.type(LongTensor)
        m_t, i_t = target.max(1)
        return int(
            sum(output.gt(0.5).type(LongTensor).ne(i_t.unsqueeze(-1)).type(FloatTensor)))  # 0.5 because of Sigmoid


# Compute train_loss train_acc val_acc and val_loss separately to use them as statistics
def compute_history_f(history, model, sum_loss, train_input, train_target, val_input, val_target, criterion):
    val_acc, val_loss = compute_test_loss_accuracy(model, val_input, val_target, criterion)
    train_acc, train_loss = compute_test_loss_accuracy(model, train_input, train_target, criterion)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    return history


# Compute loss and accuracy of a specific dataset
def compute_test_loss_accuracy(model, t_input, t_target, criterion):
    nb_samples = t_input.size(0)
    output = model.forward(t_input)
    loss = criterion.apply(output, t_target)
    _, i_o = output.max(1)
    _, i_t = t_target.max(1)
    nb_errors = sum(i_o.ne(i_t))

    return 1 - nb_errors / float(nb_samples), loss


# Normalize data wrt their mean and std
def standardization(train_input, val_input, test_input, std, mean):
    train_input -= mean
    val_input -= mean
    test_input -= mean

    train_input /= std
    val_input /= std
    test_input /= std
    return train_input, val_input, test_input


# Revert standardization process
def add_std_mean(param, std, mean):
    new_param = param.clone()
    return new_param * std + mean


# Return correct indexes, both for [0, 1] and [1, 0] labels, and mispredicted indexes
def get_correct_indexes(model, t_input, t_target):
    _, correct = model.forward(t_input).max(1)
    _, correct_index = t_target.max(1)
    blue = correct_index.ne(correct).type(LongTensor)
    opposite = (LongTensor(correct.shape).fill_(1) - correct).sub_(blue).clamp(min=0).type(ByteTensor)
    correct = correct.sub_(blue).clamp(min=0).type(ByteTensor)
    return correct, opposite, blue.type(ByteTensor)


# SUPPORT FUNCTIONS RELATED WITH PLOTS

# Plot points of a given dataset in two different colors. Only 2 different classes permitted
def plot_points(test_target, input, two_out_list, one_out_list, title='', fig=None, ax=None):
    ax.clear()
    if len(test_target.shape) > 1:
        color = ['red' if int(i_o) == 0 else 'green' for i_o in two_out_list]  # two output
    else:
        color = ['red' if l == 0 else 'green' for l in one_out_list]
    ax.scatter(input[:, 0], input[:, 1], color=color)
    ax.set_title(title)
    fig.canvas.draw()
    return fig


# Plot all the points in blue if they are mispredicted by the net, in the right color otherwise
def plot_final_points(model, t_input_plot, t_input, t_target, title, fig, ax, legend = True):
    # blue = mispredicted target, green = correct positive target, red = correct negative target
    green_ind, red_ind, blue_ind = get_correct_indexes(model, t_input, t_target)

    ax.plot(t_input_plot[:, 0][green_ind].numpy(), t_input_plot[:, 1][green_ind].numpy(), 'go', label='Correct +')
    ax.plot(t_input_plot[:, 0][red_ind].numpy(), t_input_plot[:, 1][red_ind].numpy(), 'ro', label='Correct -')
    ax.plot(t_input_plot[:, 0][blue_ind].numpy(), t_input_plot[:, 1][blue_ind].numpy(), 'bo', label='Mispredicted')

    ax.set_title(title)
    if legend:
        ax.legend()
    fig.canvas.draw()
    return fig, ax


# Initialize plot for real time visualization
def plot_initialization(std, mean):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()
    nb_of_xs = 200
    xs1 = np.linspace(-2, 2, num=nb_of_xs)
    xs2 = np.linspace(-2, 2, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid
    ex = FloatTensor(len(xx) * len(yy), 2).fill_(0)
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            ex[nb_of_xs * i + j, 0] = xx[i, j]
            ex[nb_of_xs * i + j, 1] = yy[i, j]
    xx = xx * std[0] + mean[0]
    yy = yy * std[1] + mean[1]
    return ax, xx, yy, ex, fig


# Update real time visualization, compute prediction for all points in the [0, 1]^2 plane
def real_time_plot(model, ex, xx, yy, ax, fig, val_input_plot, val_input, val_target):
    classification_plane = model.forward(ex).max(1)[1].view(200, 200).numpy()
    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.50),
        colorConverter.to_rgba('g', alpha=0.50)])
    ax.contourf(xx, yy, classification_plane, cmap=cmap)

    fig, ax = plot_final_points(model, val_input_plot, val_input, val_target, 'Dynamic results on validation dataset',
                                fig, ax)

    return ax, fig


# Finalize loss/accuracy plot
def finalize_plot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)


# Create figure and subplots for loss/accuracy plot
def prepare_plot(title, xlabel):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.suptitle(title)
    ax1.set_ylabel('MSE Loss')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2


# Add values to loss/accuracy plot
def print_save_history(history, title):
    fig, ax1, ax2 = prepare_plot(title, 'epoch')
    ax1.plot(history['train_loss'], label="training")
    ax1.plot(history['val_loss'], label="validation")
    ax2.plot(history['train_acc'], label="training")
    ax2.plot(history['val_acc'], label="validation")
    finalize_plot(fig, ax1, ax2)

    fig.canvas.draw()
