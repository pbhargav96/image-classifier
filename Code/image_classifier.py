from sys import argv
from utils.data_prep import PreProcessing
from utils.classifier import Classifier

script, location = argv


if __name__ == '__main__':
    pre_processing = PreProcessing(location)
    pre_processing.restructure_dataset()
    classifier = Classifier(location)
    classifier.run_test_harness()
