from glob import glob
import keras.preprocessing
import matplotlib.pyplot as plt
from utils import constants
import os


class PreProcessing:
    """
    Class to pre-process the dataset before feeding it to the neural network
    """
    def __init__(self, dataset_location):
        self.dataset_location = str(dataset_location)

    def explore_dataset(self):
        """
        Method to explore the dataset
        :return: nothing
        """
        # Visualize both cats and dogs
        for i in range(2):
            # View first 9 images of each type of animal
            for j in range(9):
                if i:
                    plt.figure(i + 1)
                    file_name = self.dataset_location + 'dog.' + str(j) + '.jpg'
                else:
                    plt.figure(i + 1)
                    file_name = self.dataset_location + 'cat.' + str(j) + '.jpg'
                # Read current image
                img = plt.imread(file_name)
                # Print its shape: width x height
                print(file_name, img.shape)
                # Add sub-plot region
                plt.subplot(330 + j + 1)
                # Add current image in the sub-plot
                plt.imshow(img)
            # Show each type of animal in the dataset
            plt.show()

    def extract_locations(self):
        """
        Function to get locations of all JPEG images from the given folder
        :return: a list containing proper location of all JPEG images
        """
        # Define an empty list to store file locations
        filename_list = []
        # Add file locations to the list
        for filename in glob(self.dataset_location + '*.jpg'):
            filename_list.append(filename)
        return filename_list

    def restructure_dataset(self):
        if not os.path.exists(constants.SAVE_LOCATION):
            print('Pre-processing....')
            file_names = self.extract_locations()
            os.mkdir('../train/')
            os.mkdir('../train/dogs/')
            os.mkdir('../train/cats/')
            for file in file_names:
                file_name = file.split('/')[-1]
                # Load image using keras, resize it, and convert into numpy array
                img = keras.preprocessing.image.load_img(file, target_size=(224, 224))
                img = keras.preprocessing.image.img_to_array(img)
                if 'dog' in file_name:
                    keras.preprocessing.image.save_img(constants.SAVE_LOCATION + 'dogs/' + file_name, img)
                elif 'cat' in file_name:
                    keras.preprocessing.image.save_img(constants.SAVE_LOCATION + 'cats/' + file_name, img)
        else:
            print('No pre-processing required')
