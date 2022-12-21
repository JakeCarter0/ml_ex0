import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, GlobalAvgPool2D, Flatten





# #sequential is rarely used, use functional instead
# seq_model = tensorflow.keras.Sequential(
#     [
#         tensorflow.keras.layers.Input(shape=(28,28,1)), #input layer. grayscale 28x28 (Needs extra dimension for batches(don't worry about this, but it matters for conv2D))

#         tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu"), #32 3x3 filters checked, relu used [change hyperparameters to experamentally search for success]
#         tensorflow.keras.layers.MaxPool2D(),#keeps max value for a window (2x2 by default)
#         tensorflow.keras.layers.BatchNormalization(),

#         Conv2D(128, (3,3), activation="relu"),#layer 2
#         MaxPool2D(),#keeps max value for a window (2x2 by default)
#         BatchNormalization(),

#         tensorflow.keras.layers.GlobalAvgPool2D(), #takes output from previous layer and computes average  
#         tensorflow.keras.layers.Dense(64, activation="relu"),
#         Dense(10, activation="softmax") #output layer. 10 because there are 10 different types of images. softmax because we want probability it matches
#         #input and output layer sizes are important (must match sizes), others are less important
#     ]
# )

#Functional approach:
def functional_model():

    #functional approach is much more flexible as parameters can be passed as inputs

    my_input = Input(shape=(28,28,1))

    x = Conv2D(32, (3,3), activation="relu")(my_input)
    x = MaxPool2D(3)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(10, activation="relu")(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model

def test():
    inputs = Input(shape=(784,))
    dense = Dense(64, activation="relu")
    x = dense(inputs)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10)(x)
    model = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model





#Class approach:
class MyCustomModel(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2D(32, (3,3), activation="relu"),
        self.conv2 = Conv2D(64, (3,3), activation="relu"),
        self.maxpool1 = MaxPool2D(),
        self.batchnorm1 = tensorflow.keras.layers.BatchNormalization(),

        self.conv3 = Conv2D(128, (3,3), activation="relu"),
        self.maxpool2 = MaxPool2D(),
        self.batchnorm2 = tensorflow.keras.layers.BatchNormalization(),

        self.globalavgpool1 = GlobalAvgPool2D(),
        self.dense1 = Dense(64, activation="relu"),
        self.dense2 = Dense(10, activation="softmax")


    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def streetsigns_model(num_classes=10):

    my_input = Input(shape = (60,60,3)) #approximate average size of the sign images

    x = Conv2D(32, (3,3), activation="relu")(my_input)
    x = MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    # x = GlobalAvgPool2D()(x), #takes output from previous layer and computes average
    x = Flatten()(x) #flattens dimension (4x4 -> 16)
    x = Dense(128, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    return tensorflow.keras.Model(inputs=my_input, outputs=x)


if __name__ == "__main__":

    model = test()
    model.summary()
    moddel = tensorflow.keras.preprocessing.image.ImageDataGenerator()