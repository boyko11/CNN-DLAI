import numpy as np
import tf_cnn_util, data_service, plot_service

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = data_service.load_dataset()

plot_service.plot_training_image(img_idx=8, x_train=X_train_orig, y_train=Y_train_orig)

X_train, Y_train, X_test, Y_test = \
    data_service.preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)


learning_rate = 0.009
num_epochs = 200
minibatch_size = 64

train_accuracy, test_accuracy, _, costs = tf_cnn_util.model(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate,
          num_epochs=num_epochs, minibatch_size=minibatch_size, print_cost=True)


print('Train/Test accuracy: {0}/{1}'.format(train_accuracy, test_accuracy))
plot_service.plot_lerning_curve_for_learning_rate(learning_rate, costs)




