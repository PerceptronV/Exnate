import tensorflow as tf


class RNNPlex(tf.keras.layers.Layer):
    def __init__(self, input_shape, ret_sqn, rnn_units):
        super(RNNPlex, self).__init__(name='RNNPlex')
        self.inp_shape = input_shape

        self.gru1 = tf.keras.layers.GRU(
            rnn_units, input_shape=input_shape,
            return_sequences=True
        )
        self.gru2 = tf.keras.layers.GRU(
            rnn_units, input_shape=input_shape,
            return_sequences=True
        )
        self.bn = tf.keras.layers.BatchNormalization()

        self.bidir = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                rnn_units, return_sequences=ret_sqn
            ), merge_mode='mul'
        )

    def call(self, inputs):
        x1 = self.gru1(inputs)
        x2 = self.gru2(inputs)
        x = self.bn(x1 * x2)
        x = self.bidir(x)
        return x

    def functional(self):
        inputs = tf.keras.Input(self.inp_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs, outputs, name=self.name)


class ForecastModel(tf.keras.Model):
    def __init__(self, input_shape, residual_shape, output_shape,
                 ret_sqn, rnn_units, name='ForecastModel'):

        super(ForecastModel, self).__init__(name=name)
        self.inp_shape = input_shape
        self.res_shape = residual_shape
        self.out_shape = output_shape
        self.out_dim = output_shape[-2] * output_shape[-1]

        self.rnn = RNNPlex(input_shape, ret_sqn, rnn_units)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(
            rnn_units, activation='tanh'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.dense2 = tf.keras.layers.Dense(
            int(rnn_units / 2), activation='tanh'
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.logits = tf.keras.layers.Dense(
            self.out_dim, kernel_initializer=tf.initializers.zeros
        )
        self.reshape = tf.keras.layers.Reshape(
            self.out_shape, input_shape=(self.out_dim,)
        )

    def call(self, inputs, res, func=False):
        if func:
            x = self.rnn.functional()(inputs)
        else:
            x = self.rnn(inputs)
        x = self.bn1(x)

        x = self.bn2(self.dense1(x))
        x = self.bn3(self.dense2(x))

        x = self.reshape(self.logits(x))
        return res + x

    def functional(self):
        inputs = tf.keras.Input(self.inp_shape)
        res = tf.keras.Input(self.res_shape)
        outputs = self.call(inputs, res, func=True)
        return tf.keras.Model([inputs, res], outputs, name=self.name)
