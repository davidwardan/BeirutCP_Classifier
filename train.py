from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras.metrics

from utils import utils

import pandas as pd

import random
import os

from SwinT import SwinTransformer
from config import Config


def main():
    # set seed for reproducibility
    random.seed(235)

    # define configuration
    config = Config()

    # get input, normalize and print shapes
    x_train = utils.load_data(config.in_dir + '/train/x_train.npy')
    y_train = utils.load_data(config.in_dir + '/train/y_train.npy')

    x_train = utils.norm_image(x_train)
    y_train = utils.to_categorical(y_train, config.num_classes)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")

    # if validation input is available
    if config.val == 1:
        x_val = utils.load_data(config.in_dir + '/val/x_val.npy')
        y_val = utils.load_data(config.in_dir + '/val/y_val.npy')
        x_val = utils.norm_image(x_val)
        y_val = utils.to_categorical(y_val, config.num_classes)
        print(f"x_val shape: {x_val.shape} - y_val shape: {y_val.shape}")

    print('Input stored...')

    # Build model
    inputs = Input(config.input_shape)

    # Define a base model
    if config.transfer_learning == 1:
        if config.image_size != 224:
            raise ValueError('Invalid image size for transfer learning')
        x = SwinTransformer('swin_base_224', include_top=False, pretrained=True)(inputs)
        x.trainable = True
    else:
        x = SwinTransformer('swin_base_224', include_top=False, pretrained=False)(inputs)
        x.trainable = True

    # FC layers
    '''
    The following code is an example of how to add a fully connected layer to the model.
    You can add more layers, change the number of units, activation functions, etc.
    '''
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(64, activation='relu')(x)

    # Output
    out = Dense(units=config.num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=out)
    if config.optimizer == 'adam':
        optimizer = Adam(learning_rate=config.lr_max)
    elif config.optimizer == 'sgd':
        optimizer = SGD(learning_rate=config.lr_max)
    else:
        raise ValueError('Invalid optimizer')

    model.compile(optimizer=optimizer, loss=config.loss, metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"),
                                                                  keras.metrics.Precision(), keras.metrics.Recall()])
    print('Model ready to train...')

    # Fit the model
    if config.val == 1:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                      patience=3, min_lr=config.lr_min)

        early_stop = EarlyStopping(monitor='val_accuracy', patience=config.ealry_stop_patience)

        history = model.fit(x_train, y_train, batch_size=config.batch_size, shuffle=True,
                            epochs=config.num_epochs, verbose=2, callbacks=[early_stop, reduce_lr],
                            validation_data=(x_val, y_val))
    else:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,
                                      patience=3, min_lr=config.lr_min)

        early_stop = EarlyStopping(monitor='accuracy', patience=config.ealry_stop_patience)

        history = model.fit(x_train, y_train, batch_size=config.batch_size, shuffle=True,
                            epochs=config.num_epochs, verbose=2, callbacks=[early_stop, reduce_lr])

    # save history
    print("History saved...")
    df = pd.DataFrame(history.history)
    df.to_csv(config.out_dir + '/history.csv')

    # save model
    keras.saving.save_model(model, config.out_dir + '/SwinT')


if __name__ == "__main__":
    main()
