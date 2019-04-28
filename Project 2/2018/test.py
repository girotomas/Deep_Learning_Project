import time

import matplotlib.pyplot as plt

import activations as F
import loss
import modules as nn
import optimizer
from Utils.datasets import generate_disc_set
from Utils.support_functions import compute_nb_errors, standardization, plot_points, plot_initialization, \
    real_time_plot, compute_history_f, print_save_history, plot_final_points

# Generate all needed datasets
center = (0.5, 0.5)
train_input, train_target = generate_disc_set(1000, center)  # Used to train the model
val_input, val_target = generate_disc_set(1000, center)  # Used to check performances during training
test_input, test_target = generate_disc_set(1000, center)  # Used to check final accuracy

# Data standardization
mean = train_input.mean(0)
std = train_input.std(0)
val_input_plot = val_input.clone()
test_input_plot = test_input.clone()
train_input, val_input, test_input = standardization(train_input, val_input, test_input, std, mean)

# plot learning objective
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()
fig = plot_points(test_target, test_input_plot, test_target.max(1)[1], test_target, 'Test dataset', fig, ax)


def train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epochs, criterion,
                early_stopping=False, compute_history=False):
    # Initialization for early stopping
    previous_loss = 1e5
    patience_count = 0
    patience = 3
    mini_batch_size_initial = mini_batch_size

    # history initialization
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # initialization for real-time surface
    ax_rt, xx, yy, ex, fig_rt = plot_initialization(std, mean)

    for e in range(epochs):
        sum_loss = 0
        mini_batch_size = mini_batch_size_initial
        for b in range(0, train_input.size(0), mini_batch_size):
            if (train_input.size(0) - b) < mini_batch_size:
                mini_batch_size = train_input.size(0) - b

            optim.zero_grad(model)  # Reset dw and db for each layer

            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.apply(output, train_target.narrow(0, b, mini_batch_size))
            model.backward(criterion, train_target.narrow(0, b, mini_batch_size), mini_batch_size)

            optim.step(model)
            sum_loss = sum_loss + loss

        # plot semi-real time graph every 30 epochs but always before the 80th to see changes
        if not e % 30 or e < 80:
            ax_rt.clear()
            ax_rt, fig_rt = real_time_plot(model, ex, xx, yy, ax_rt, fig_rt, val_input_plot, val_input, val_target)

        # compute loss and accuracy for both test and validation. Can be useful during the development process
        if compute_history:
            history = compute_history_f(history, model, sum_loss, train_input, train_target, val_input, val_target,
                                        criterion)

        # print epoch, loss and validation_accuracy every 100 epochs
        if not e % 10:
            if compute_history:
                print("{0:d}".format(e),
                      "train_loss: {0:.4f}".format(history['train_loss'][-1]),
                      "val_loss: {0:.4f}".format(history['val_loss'][-1]),
                      "train_acc: {0:.3f}".format(history['train_acc'][-1]),
                      "val_acc: {0:.3f}".format(history['val_acc'][-1]))
            else:
                nb_errors = compute_nb_errors(model, val_input, val_target)
                print("epoch: ", e, " train_loss: ", sum_loss, ' val_accuracy:', 1 - (nb_errors / val_target.shape[0]))

        # early stopping implementation
        if early_stopping:
            if sum_loss > previous_loss:
                patience_count += 1
                if patience_count > patience:
                    return
            else:
                patience_count = 0

            previous_loss = sum_loss
    return history


# Training parameters
lr = 1e-4
optim = optimizer.SGD(lr)
mini_batch_size = 100
epoch = 2000
criterion = loss.MSELoss()

# Model creation
model = nn.Sequential(
    nn.Dense(2, 25, F.ReLU()),
    nn.Dense(25, 25, F.ReLU()),
    nn.Dense(25, 25, F.ReLU()),
    nn.Dense(25, 2, F.Sigmoid())
)

# Train the model
print(train_input.shape)
compute_history = False
history = train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion,
                      True, compute_history)
plt.show()

if compute_history:
    print_save_history(history, 'Training and Validation')

# plot final result on test dataset: accuracy and graph
nb_errors = compute_nb_errors(model, test_input, test_target)
test_accuracy = 1 - nb_errors / test_input.shape[0]
print('Test accuracy:', test_accuracy)
plot_final_points(model, test_input_plot, test_input, test_target, 'Final Result on Test dataset - Accuracy: ' + str(test_accuracy), fig, ax, False)

time.sleep(20)
