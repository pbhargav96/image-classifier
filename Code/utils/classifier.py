import matplotlib.pyplot as plt
import keras


# Define CNN model of 3 layers
def define_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                                  input_shape=(224, 224, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # compile model
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
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


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = data_gen.flow_from_directory('../train/', class_mode='binary', batch_size=64,
                                            target_size=(224, 224))
    test_it = data_gen.flow_from_directory('../dogs-vs-cats/test1/', class_mode='binary', batch_size=64,
                                           target_size=(224, 224))
    # Define callbacks
    checkpoint = keras.callbacks.ModelCheckpoint("first_model.h5", monitor='val_acc', verbose=0, save_best_only=True,
                                                 save_weights_only=False, mode='auto',
                                                 period=1)
    early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                                  validation_steps=len(test_it), epochs=5, verbose=1, callbacks=[checkpoint, early])
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
