import numpy as np
import matplotlib.pyplot as plt


def plot_training_image(img_idx, x_train, y_train):

    plt.imshow(x_train[img_idx])
    plt.show()
    print("y = " + str(np.squeeze(y_train[:, img_idx])))

def plot_lerning_curve_for_learning_rate(learning_rate, costs):
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
