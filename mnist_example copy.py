import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

#Training and Evaluation : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #This doesn't:
        #Save model
        #Save best model based on best verification result
        #Use a custom dataset (uses hello world mnist)


#tensorflow.keras.Sequential approach:
#sequential is rarely used, use functional instead
seq_model = tensorflow.keras.Sequential(
    [
        tensorflow.keras.layers.Input(shape=(28,28,1)), #input layer. grayscale 28x28 (Needs extra dimension for batches(don't worry about this, but it matters for conv2D))

        tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu"), #32 3x3 filters checked, relu used [change hyperparameters to experamentally search for success]
        tensorflow.keras.layers.MaxPool2D(),#keeps max value for a window (2x2 by default)
        tensorflow.keras.layers.BatchNormalization(),

        Conv2D(128, (3,3), activation="relu"),#layer 2
        MaxPool2D(),#keeps max value for a window (2x2 by default)
        BatchNormalization(),

        tensorflow.keras.layers.GlobalAvgPool2D(), #takes output from previous layer and computes average  
        tensorflow.keras.layers.Dense(64, activation="relu"),
        Dense(10, activation="softmax") #output layer. 10 because there are 10 different types of images. softmax because we want probability it matches
        #input and output layer sizes are important (must match sizes), others are less important
    ]
)
#

#Functional approach:
def functional_model():

    #functional approach is much more flexible as parameters can be passed as inputs

    my_input = Input(shape=(28,28,1)),

    x = Conv2D(32, (3,3), activation="relu")(my_input),
    x = MaxPool2D()(x),
    x = BatchNormalization()(x),

    x = Conv2D(128, (3,3), activation="relu")(x),
    x = MaxPool2D()(x),
    x = BatchNormalization()(x),

    x = GlobalAvgPool2D()(x),
    x = Dense(64, activation="relu")(x),
    x = Dense(10, activation="softmax")(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model

#Class approach:
class MyCustomModel(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2D(32, (3,3), activation="relu"),
        self.conv2 = Conv2D(64, (3,3), activation="relu"),
        self.maxpool1 = MaxPool2D(),
        self.batchnorm1 = BatchNormalization(),

        self.conv3 = Conv2D(128, (3,3), activation="relu"),
        self.maxpool2 = MaxPool2D(),
        self.batchnorm2 = BatchNormalization(),

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


def display_some_examples(examples, labels):

    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.imshow(img, cmap = "gray") #display as grayscale
    plt.show()



if __name__=='__main__': # only runs this code if this file is called directly (not imported)
    
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data() #gets images of drawn numbers

    print("x_train.shape = ", x_train.shape) #image
    print("y_train.shape = ", y_train.shape) #number corresponding to image
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)


    # display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255 #normalizes (helps NN)
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=3) #makes shape match shape from input EG(28,28,1) instead of (28,28)
    x_test = np.expand_dims(x_test, axis=-1) #-1 adds extra dimension to end, so same as 3


    


    #     #Sequential:
    # seq_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics="accuracy")
    #     #optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers,
    #     #popular classification loss function is crossentropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses
    #         #categorical_crossentropy expects one_hot representation of labels, use sparse_categorical_crossentropy if not one_hot (see documentation)
    #             #label : 2 => one hot encoding : [0,0,1,0,0,0,0,0,0,0]
    #             #to convert to one_hot with 10 catagories: y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    #     #metrics determines how prediction is graded (88 accuracy is 88% accuracy)

    #     #Model training:
    # seq_model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    #     #batch is how many images are considered at a time for preformance enhancement
    #     #epochs is how many times the model sees the entire dataset (too many gets too trained for the specific dataset, too few isn't enough data)
    #     #validation_spit determines how many of train set is used to initially test model. (shows how well it preforms on training set)(80% used for training, 20% used for validatioon)

    #     #Evaluation on test set
    # seq_model.evaluate(x_test,y_test,batch_size=64)
    
    #     #Functional approach:
    # model = functional_model()
    # model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics="accuracy")
    #     #optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers,
    #     #popular classification loss function is crossentropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses
    #         #categorical_crossentropy expects one_hot representation of labels, use sparse_categorical_crossentropy if not one_hot (see documentation)
    #             #label : 2 => one hot encoding : [0,0,1,0,0,0,0,0,0,0]
    #             #to convert to one_hot with 10 catagories: y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    #     #metrics determines how prediction is graded (88 accuracy is 88% accuracy)

    #     #Model training:
    # model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    #     #batch is how many images are considered at a time for preformance enhancement
    #     #epochs is how many times the model sees the entire dataset (too many gets too trained for the specific dataset, too few isn't enough data)
    #     #validation_spit determines how many of train set is used to initially test model. (shows how well it preforms on training set)(80% used for training, 20% used for validatioon)

    #     #Evaluation on test set
    # model.evaluate(x_test,y_test,batch_size=64)

        #Class approach:
    model = MyCustomModel()
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics="accuracy")
        #optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers,
        #popular classification loss function is crossentropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses
            #categorical_crossentropy expects one_hot representation of labels, use sparse_categorical_crossentropy if not one_hot (see documentation)
                #label : 2 => one hot encoding : [0,0,1,0,0,0,0,0,0,0]
                #to convert to one_hot with 10 catagories: y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        #metrics determines how prediction is graded (88 accuracy is 88% accuracy)

        #Model training:
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
        #batch is how many images are considered at a time for preformance enhancement
        #epochs is how many times the model sees the entire dataset (too many gets too trained for the specific dataset, too few isn't enough data)
        #validation_spit determines how many of train set is used to initially test model. (shows how well it preforms on training set)(80% used for training, 20% used for validatioon)

        #Evaluation on test set
    model.evaluate(x_test,y_test,batch_size=64)



    #Additonal stuff:~~~~~~~~~~~~~~~~~~~~~~~~~~




