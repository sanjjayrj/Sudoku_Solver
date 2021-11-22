from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
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
        model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu",
            input_shape=(height, width, channels)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2), strides=2, padding="same"))
        model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=num_categories, activation="softmax"))

        print(model.summary())
        print(model.compile(loss='categorical_crossentropy', metrics=['accuracy']))

        return model

"""    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
    args = vars(ap.parse_args())
"""


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
    sudoku_model = sudoku_model()
    model = sudoku_model.build(28, 28, 1, num_categories)
    model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid))
    model.save("sudoku_model", save_format="h5")