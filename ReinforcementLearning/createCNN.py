from tensorflow import keras

if __name__ == "__main__":

    model = keras.Sequential()
    model.add(keras.Input(shape=(4, 4, 1), name='board'))
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='conv1'))
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='conv2'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='1'))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='2'))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='3'))
    model.add(keras.layers.Dense(1, activation='linear', name='output'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])

    model.save('initCNN.h5', overwrite=True)
