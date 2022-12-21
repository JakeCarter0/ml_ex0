

import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from my_utils import split_data, order_test_set, create_generators
from deeplearningmodels import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow


if __name__ == "__main__":

    # path_to_data = "C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\archive\\Train" #2 backslashes prevents unix/linux errors
    # path_to_save_train = "C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\training_data\\train"
    # path_to_save_val = "C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\training_data\\val"
    # split_data(path_to_data=path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val, split_size=0.1)


    
    # path_to_images = "C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\archive\\Test"
    # path_to_csv = "C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\archive\\Test.csv"
    # order_test_set(path_to_images=path_to_images, path_to_csv=path_to_csv)

    path_to_train = "./training_data/train"
    path_to_val = "./training_data/val"
    path_to_test = "./archive/Test"
    batch_size = 64
    epochs = 15
    lr=0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size=batch_size, train_data_path=path_to_train,val_data_path=path_to_val, test_data_path=path_to_test)

    TRAIN = True
    TEST = False

    
    if TRAIN:


        path_to_save_model = "./Models"
        ckpt_saver = ModelCheckpoint( #saves best model
            path_to_save_model,
            monitor="accuracy", #grades models on accuracy
            mode="max", #saves max accuracy (use min for val_loss)
            save_best_only=True,
            save_freq="epoch", #saves model after each epoch
            verbose=1 #prints status to console
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10 #if val_accuaracy doesnt imporve over the last 10 epochs then quit

        )


        model = streetsigns_model(train_generator.num_classes)

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver,early_stop] #saves models
            )


    if TEST:
        model = tensorflow.keras.models.load_model("./Models")
        model.summary()
        
        model.evaluate(val_generator)

        model.evaluate(test_generator)

    


