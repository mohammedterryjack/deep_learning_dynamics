############   NATIVE IMPORTS  ###########################
from typing import NewType, List
############ INSTALLED IMPORTS ###########################
from numpy import array
############   LOCAL IMPORTS   ###########################
##########################################################

Vectors = NewType('Vectors',List[array])
Labels = NewType('Labels',List[str])

class DataFrameNames:
    X_COORDINATE = "x coordinate"
    Y_COORDINATE = "y coordinate"
    VECTOR = "vector"
    NETWORK_NAME = "neural network"
    NETWORK_SCORE = "score"
    COLOUR = "colour"