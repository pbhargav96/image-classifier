import matplotlib.pyplot as plt
import keras
from utils import constants


class Classifier:
    """
    Class to train a model for classification
    """
    def __init__(self, dataset_location):
        """
        Method to initialize classifier class
        :param dataset_location: location of the dataset
        """
        self.dataset_location = str(dataset_location)

    @staticmethod
    def define_model():
        """
        Define CNN model of 3 layers
        :return: model
        """

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same",
                                      activation="relu"))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=4096, activation="relu"))
        model.add(keras.layers.Dense(units=4096, activation="relu"))
        model.add(keras.layers.Dense(units=1, activation="softmax"))
        opt = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        return model

    @staticmethod
    def summarize_diagnostics(history):
        """
        Method to plot diagnostic learning curves
        :param history: a data structure that stores training and testing metrics
        :return: nothing
        """
        # Print training loss and accuracy
        for i in range(len(history.history['loss'])):
            print('Epoch:', i, '| Loss:', history.history['loss'][i], '| Train Accuracy:',
                  100 * history.history['accuracy'][i])
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        # show the plot
        plt.legend()
        plt.show()
        # save plot to file
        # filename = sys.argv[0].split('/')[-1]
        # plt.savefig(filename + '_plot.png')
        # plt.close()

    def run_test_harness(self):
        """
        Method to train the model and test it
        :param self:
        :return:
        """
        # define model
        model = self.define_model()
        # create data generator
        # data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
        train_gen = keras.preprocessing.image.ImageDataGenerator()
        test_gen = keras.preprocessing.image.ImageDataGenerator()
        # prepare iterators
        train_it = train_gen.flow_from_directory(constants.SAVE_LOCATION, target_size=(224, 224))
        test_it = test_gen.flow_from_directory(self.dataset_location[:-6] + 'test/', target_size=(224, 224))
        # fit model
        history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, shuffle=True,
                                      validation_steps=len(test_it), epochs=5, verbose=1)
        # evaluate model
        _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history)
