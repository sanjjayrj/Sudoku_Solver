from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization
)

class sudoku_model:
    @staticmethod
    # this is the dimensions of the digits
    def build(height, width, channels, classes):
        model = Sequential()
        model.add(Conv2D(50, (5, 5), strides=1, padding="same", activation="relu",
            input_shape=(height, width, channels)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2), strides=2, padding="same"))
        model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2), strides=2, padding="same" ))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(units=num_categories, activation="softmax"))

        print(model.summary())
        print(model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']))

        return model


if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data() 
    x_train = x_train.reshape(-1,28,28,1)/255
    x_valid = x_valid.reshape(-1,28,28,1)/255
    print(x_train.shape)
    # categorical encoding for better classification
    # since there are 10 digits
    num_categories = 10
    y_train = keras.utils.to_categorical(y_train, num_categories)
    y_valid = keras.utils.to_categorical(y_valid, num_categories)

    # we are augmenting the data now
    # datagen = ImageDataGenerator(
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=False,
    #     vertical_flip=False
    #     )
    batch = 32
    # datagen.fit(x_train)
    # image_iter = datagen.flow(x_train, y_train, batch_size = batch)
    sudoku_model = sudoku_model()
    model = sudoku_model.build(28, 28, 1, num_categories)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=10,
        verbose=1,
        batch_size=batch
    )
    model.save("sudoku_model.h5", save_format="h5")