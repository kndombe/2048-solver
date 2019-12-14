import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":

    s = keras.Input(shape=(16,), dtype=tf.int64, name='board')
    a = keras.Input(shape=(1,), name='direction')
    x = keras.layers.Dense(128, activation='relu', name='s1',
                           kernel_regularizer=keras.regularizers.l2(l=0.01))(s)
    x = keras.layers.Dense(64, activation='relu', name='s2',
                           kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
    x = keras.layers.concatenate([x, a], name='Concact')
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='1')(x)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='2')(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='3')(x)
    outputs = keras.layers.Dense(1, activation='linear', kernel_regularizer=keras.regularizers.l2(
        l=0.01), bias_regularizer=keras.regularizers.l2(l=0.01), name='output')(x)
    model = keras.Model(inputs=[s, a], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(
    ), loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])

    model.save('initFC', overwrite=True)
