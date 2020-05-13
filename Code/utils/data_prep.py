import matplotlib.pyplot as plt


def explore_dataset(location):
    """
    Function to explore the dataset
    :param location: location of the dataset relative to the project directory
    :return: nothing
    """
    # Visualize both cats and dogs
    for i in range(2):
        # View first 9 images of each type of animal
        for j in range(9):
            if i:
                plt.figure(i + 1)
                file_name = location + 'dog.' + str(j) + '.jpg'
            else:
                plt.figure(i + 1)
                file_name = location + 'cat.' + str(j) + '.jpg'
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


if __name__ == '__main__':
    explore_dataset('../dogs-vs-cats/train/')
