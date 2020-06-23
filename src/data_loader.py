############   NATIVE IMPORTS  ###########################
from typing import Tuple
############ INSTALLED IMPORTS ###########################
from sklearn.datasets import fetch_openml
from numpy import array
############   LOCAL IMPORTS   ###########################
from data_types import Vectors, Labels
##########################################################

class DataLoader:

    @staticmethod
    def load(dataset:str) -> Tuple[Vectors,Labels,Labels]:
        if dataset == "mnist":
            return DataLoader.mnist()
        if dataset == "omniglot":
            return DataLoader.omniglot()
        return [],[],[]

    @staticmethod
    def mnist() -> Tuple[Vectors,Labels,Labels]:
        """ returns: training_inputs, training_outputs, classes for MNIST"""
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        return x/255., y, list(map(str,range(10)))
    
    @staticmethod
    def omniglot() -> Tuple[Vectors,Labels,Labels]:
        """ returns: training_inputs, training_outputs, classes for OMNIGLOT"""
        #TODO this one
        return [],[],[]