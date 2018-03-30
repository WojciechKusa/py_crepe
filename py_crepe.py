"""Crepe class module."""
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

class Crepe(object):
    """Char CNN model."""
    def __init__(self):
        super(Crepe, self).__init__()


    @staticmethod
    def build(filter_kernels, dense_layer_units, maxlen, vocab_size, filters,
              output_size, loss='categorical_crossentropy', optimizer='adam'):
        """builds model."""
        #Define what the input shape looks like
        inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

        #All the convolutional layers...
        conv1 = Conv1D(filters=filters, kernel_size=filter_kernels[0],
                             padding='valid', activation='relu',
                             input_shape=(maxlen, vocab_size))(inputs)
        conv1 = MaxPooling1D(pool_size=3)(conv1)

        conv2 = Conv1D(filters=filters, kernel_size=filter_kernels[1],
                              padding='valid', activation='relu')(conv1)
        conv2 = MaxPooling1D(pool_size=3)(conv2)

        conv3 = Conv1D(filters=filters, kernel_size=filter_kernels[2],
                       padding='valid', activation='relu')(conv2)

        conv4 = Conv1D(filters=filters, kernel_size=filter_kernels[3],
                       padding='valid', activation='relu')(conv3)

        conv5 = Conv1D(filters=filters, kernel_size=filter_kernels[4],
                       padding='valid', activation='relu')(conv4)

        conv6 = Conv1D(filters=filters, kernel_size=filter_kernels[5],
                       padding='valid', activation='relu')(conv5)
        conv6 = MaxPooling1D(pool_size=3)(conv6)
        conv6 = Flatten()(conv6)

        #Two dense layers with dropout of .5
        z = Dropout(0.5)(Dense(dense_layer_units, activation='relu')(conv6))
        z = Dropout(0.5)(Dense(dense_layer_units, activation='relu')(z))

        #Output dense layer with softmax activation
        output = Dense(output_size, activation='softmax', name='output')(z)

        model = Model(inputs=inputs, outputs=output)

        if optimizer == 'sgd':
            optimizer = SGD(lr=0.01, momentum=0.9)

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model
