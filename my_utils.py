import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
import tensorflow


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


def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    """
    Function that loads sign data to train and validate model into seperate folders to prep for processing
    """

    folders = os.listdir(path_to_data)                                                       #returns list of folders in path_to_data

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)                                       #full_path is the path to the train number folder
        images_paths = glob.glob(os.path.join(full_path, '*.png'))                           #returns a list of all filenames that match the unix styyle of '*.png' (list of image filenames in training set)

        x_train, x_val = train_test_split(images_paths, test_size = split_size)              #splits training images into training set and validation set

        for x in x_train:                                                                    #iterates through each image in train set

            # basename = os.path.basename(x) #gives name of file without extension (.png)
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:                                                                    #iterates through each image in validation set

            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)



def order_test_set(path_to_images, path_to_csv):
    """
    Reads csv file to catagorize test images into correct folders
    """

    testset = {}

    try:
        with open(path_to_csv, "r") as csvfile:

            reader = csv.reader(csvfile, delimiter = ",")

            for i, row in enumerate(reader):

                if i==0:
                    continue
                
                img_name = row[-1].replace("Test/", "")
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)
                
                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)
    
    except:
        print("error reading csv")


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        
        rescale = 1 / 255. #rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations)
        # rotation_range = 10,
        # width_shift_range=0.1

    ) #Generate batches of tensor image data with real-time data augmentation (Outdated, see documentation)

    test_preprocessor = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        
        rescale = 1 / 255. #rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations)

    ) #Generate batches of tensor image data with real-time data augmentation (Outdated, see documentation)

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path, #search through training directory, with each folder within having a certain catagory of image already sorted
        class_mode="categorical", #must also use catagorical cross entropy loss model. Also one hot encoding
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path, 
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False, #doesn't matter for validation
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator