from tensorflow.keras.layers import (Conv2DTranspose, ConvLSTM2D, Conv2D,
                                     TimeDistributed, LayerNormalization)
from tensorflow.keras import Model


class ModelADetector(Model):
    def __init__(self):
        super(ModelADetector, self).__init__()
        # dowsample
        self.td_con_0 = TimeDistributed(Conv2D(128, (11, 11), strides=4))
        self.ln_1 = LayerNormalization()
        self.td_conv_1 = TimeDistributed(Conv2D(64, (5, 5), strides=2))
        self.ln_2 = LayerNormalization()
        # bottle neck
        self.conv_lstm_1 = ConvLSTM2D(64, (3, 3), padding='same',
                                      return_sequences=True)
        self.ln_3 = LayerNormalization()
        self.conv_lstm_2 = ConvLSTM2D(32, (3, 3), padding='same',
                                      return_sequences=True)
        self.ln_4 = LayerNormalization()
        self.conv_lstm_3 = ConvLSTM2D(64, (3, 3), padding='same',
                                      return_sequences=True)
        self.ln_5 = LayerNormalization()
        # upsammpling
        self.td_convT_1 = TimeDistributed(Conv2DTranspose(128, (5, 5),
                                                          strides=2))
        self.ln_6 = LayerNormalization()
        self.td_convT_2 = TimeDistributed(Conv2DTranspose(1, (11, 11),
                                                          activation="sigmoid",
                                                          strides=4))

    def call(self, inputs):
        x = self.td_con_0(inputs)
        x = self.ln_1(x)
        x = self.td_conv_1(x)
        x = self.ln_2(x)
        x = self.conv_lstm_1(x)
        x = self.ln_3(x)
        x = self.conv_lstm_2(x)
        x = self.ln_4(x)
        x = self.conv_lstm_3(x)
        x = self.ln_5(x)
        x = self.td_convT_1(x)
        x = self.ln_6(x)
        x = self.td_convT_2(x)

        return x
