from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import L2
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, BatchNormalization


def Over_Fitted_Model():
    model = Sequential(
        [

            Dense(100, input_shape=(784,), activation='relu',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1, seed=0),
                  name='hidden_1'),
            Dense(100, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1, seed=0),
                  name='hidden_2'),
            Dense(100, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1, seed=0),
                  name='hidden_3'),
            Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1, seed=0),
                  name='output'),
        ],
        name='Over_Fitted_Model'
    )
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def Regularized_model():
    model = Sequential(
        [
            Dropout(0.2, input_shape=(784,)),
            Dense(100, input_shape=(784,),
                  activation='relu',
                  kernel_regularizer=L2(0.00006),
                  name='hidden_1'),
            BatchNormalization(),
            Dense(100, activation='relu',
                  kernel_regularizer=L2(0.00006),
                  name='hidden_2'),
            Dropout(0.10),
            Dense(100, activation='relu',
                  kernel_regularizer=L2(0.00006),
                  name='hidden_3'),
            Dense(1, activation='sigmoid', name='output'),
        ],
        name='Regularized_Model'
    )
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model
