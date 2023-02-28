model_1 = models.Sequential(name = 'Over_Fitted_Model')
model_1.add(layers.Dense(100, input_shape = (784,),
                        activation = 'relu',
                        kernel_initializer = RandomNormal(mean=0.0, stddev=1, seed=0),
                        name = 'hidden_1'))
model_1.add(layers.Dense(100, activation = 'relu',
                        kernel_initializer = RandomNormal(mean=0.0, stddev=1, seed=0),
                        name = 'hidden_2'))
model_1.add(layers.Dense(100, activation = 'relu',
                        kernel_initializer = RandomNormal(mean=0.0, stddev=1, seed=0),
                        name='hidden_3'))
model_1.add(layers.Dense(1, activation='sigmoid',
                        kernel_initializer = RandomNormal(mean=0.0, stddev=1, seed=0),
                        name='output'))

model_1.summary()

model_2 = models.Sequential(name = 'Regularized_Model')
L2_reg = regularizers.L2(0.00006)
model_2.add(Dropout(0.2, input_shape=(784,)))
model_2.add(layers.Dense(100, input_shape = (784,),
                         activation = 'relu',
                         kernel_regularizer = L2_reg,
                         name = 'hidden_1'))
model_2.add(layers.BatchNormalization())
model_2.add(layers.Dense(100, activation = 'relu',
                         kernel_regularizer = L2_reg,
                         name='hidden_2'))
model_2.add(Dropout(0.10))
model_2.add(layers.Dense(100, activation='relu',
                         kernel_regularizer= L2_reg,
                         name='hidden_3'))
model_2.add(layers.Dense(1, activation='sigmoid', name='output'))

model_2.summary()