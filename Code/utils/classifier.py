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
        Define a CNN model
        :return: model
        """
        # Add 3 convolution layers with batch normalization and pooling
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                                      input_shape=(200, 200, 3)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        # compile model
        sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
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

    def run_test_harness(self):
        """
        Method to train the model and test it
        :param self:
        :return:
        """
        # define model
        model = self.define_model()
        # create data generator
        data_gen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        data_gen.mean = [123.68, 116.779, 103.939]
        # prepare iterators
        train_it = data_gen.flow_from_directory(constants.SAVE_LOCATION, class_mode='binary', batch_size=64,
                                                target_size=(224, 224))
        test_it = data_gen.flow_from_directory(self.dataset_location[:-6] + 'test/', class_mode='binary', batch_size=64,
                                               target_size=(224, 224))
        # fit model
        history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, shuffle=True,
                                      validation_steps=len(test_it) // 2, epochs=10, verbose=1)
        # evaluate model
        _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history)
