# This file was influenced by davidADSP on github

import config
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.models import load_model, Model
from keras import regularizers
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits
import numpy as np

class Residual_CNN:

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        self.model.fit(states=states, targets=targets
                       , epochs=epochs
                       , verbose=verbose
                       , validation_split=validation_split
                       , batch_size=batch_size)

    def save(self, version):
        # Save the model
        self.model.save('models/version' + "{0:0>4}".format(version) + '.h5')

    def load(self):
        # Load the model
        return

    def convolutional_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters
                   , kernel_size=kernel_size
                   , data_format="channels_first"
                   , padding='same'
                   , use_bias=False
                   , activation='linear'
                   , kernel_regularizer=regularizers.l2(self.reg_const))(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.convolutional_layer(input_block, filters, kernel_size)

        x = Conv2D(filters=filters
                   , kernel_size=kernel_size
                   , data_format="channels_first"
                   , padding='same'
                   , use_bias=False
                   , activation='linear'
                   , kernel_regularizer=regularizers.l2(self.reg_const))(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return x

    def policy_head(self, x):
        x = self.convolutional_layer(x, 2, (1, 1))

        x = Flatten()(x)

        x = Dense(
            self.output_dim,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='policy_head'
        )(x)

        return x

    def value_head(self, x):
        x = self.convolutional_layer(x, 1, (1, 1))

        x = Flatten()(x)

        x = Dense(
            20,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name="value_head"
        )(x)

        return x

    def build_model(self):
        print(self.input_dim)
        main_input = Input(shape=self.input_dim, name="main_input")

        x = self.convolutional_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        value_head = self.value_head(x)
        policy_head = self.policy_head(x)

        nn_model = Model(inputs=[main_input], outputs=[value_head, policy_head])

        nn_model.compile(
            loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
            optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        )

        return nn_model

    def convert_to_input(self, state):
        input_model = np.zeros((2, 19, 19))
        for i in range(state.board_size - 1, -1, -1):
            for j in range(0, state.board_size):
                this_piece = state.board.get((i, j))
                if this_piece is None:
                    input_model[0][i][j] = 0
                    input_model[1][i][j] = 0
                if this_piece == 'b':
                    input_model[0][i][j] = 1
                if this_piece == 'w':
                    input_model[1][i][j] = 1
        #print(input_model)
        input_model = np.reshape([input_model], self.input_dim)
        return input_model

    def write(self, version):
        self.model.save('models' + "{0:0>4}".format(version) + '.h5')

    def read(self, game, run_number, version):
        return load_model("/models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})


    def __init__(self, reg_const, learning_rate, input_dim,  output_dim, hidden_layers):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self.build_model()
