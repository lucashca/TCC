

def baseline_model():
    # create model
    activation = 'relu'  # or linear
    init_mode = 'uniform'
    # create model
    model = Sequential()
    model.add(Dense(8,
                    input_dim=2, kernel_initializer=init_mode,
                    activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam',
                  metrics=['mse', 'mae'])
    model.summary()
    return model
