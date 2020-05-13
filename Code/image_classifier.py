from sys import argv
from utils.data_prep import PreProcessing

script, location = argv


if __name__ == '__main__':
    pre_processing = PreProcessing(location)
    pre_processing.restructure_dataset()
