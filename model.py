import numpy as np
import pandas as pd
import tensorflow as tf
import preprocessing
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from keras.constraints import maxnorm
from tensorflow.keras import regularizers

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


def create_model(learning_rate, classification_threshold):
    model = models.Sequential()

    # Define the first hidden layer with 20 nodes.
    # model.add(tf.keras.layers.Dense(units=10,
    #                                 activation='softmax',
    #                                 name='Hidden1',
    #                                 activity_regularizer=regularizers.l2(0.01)
    #                                 ))

    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1))

    # model.add(layers.Dense(10))

    # checkpoint = ModelCheckpoint('best_model_improved.h5',
    #                              monitor='val_loss',
    #                              verbose=0,
    #                              save_best_only=True,
    #                              mode='auto')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold)])

    return model


def train_model(model, train_images, train_labels, test_images, test_labels, epochs, batch_size=None):
    # Split the dataset into features and label.

    history = model.fit(train_images, train_labels, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_data=(test_images, test_labels))

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    # for m in list_of_metrics:
    x = hist["accuracy"]
    plt.plot(epochs[1:], x[1:], label="accuracy")

    plt.show()

features, label = preprocessing.load_data()
features, label = preprocessing.shuffle_data(features, label)

# Change the arrays into numpy arrays, and reshaping them.
X_train = np.array(features[:12000]).reshape(-1, 224, 224, 3)
y_train = np.array(label[:12000])
X_test = np.array(np.array(features[12000:])).reshape(-1, 224, 224, 3)
y_test = np.array(label[12000:])

learning_rate = 0.1
epochs = 20
batch_size = 32
classification_threshold = 0.5


my_model = create_model(learning_rate, classification_threshold)
epoch, hist = train_model(my_model, X_train, y_train, X_test, y_test, epochs, batch_size)
plot_curve(epoch, hist, [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold)])