############   NATIVE IMPORTS  ###########################
from random import randint
from typing import List, Iterable
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPClassifier
from numpy import zeros
from seaborn import heatmap
from matplotlib.pyplot import show
from pandas import read_pickle, DataFrame
from pickle import dump
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
from neural_grid_search import NeuralGridSearch
from projection_models.autoencoder import AutoEncoder
##########################################################

#TODO: heatmap of single hidden layers weight space. Get accuracy for each coordinate (find corresponding weight for that coordinate and test network)
#TODO: work out which weights produced the better scores from the heatmap (i.e. convert their coordinates back to weights) and see if there is anything in common with them
#TODO: if initialised in good areas of the weight space - and see how this affects training

# INPUT_LAYER_SIZE = 784
# HIDDEN_LAYER_SIZE = 2
# OUTPUT_LAYER_SIZE = 10

# x,y,c = DataLoader.load("mnist")

# network = MLPClassifier(
#     hidden_layer_sizes= (HIDDEN_LAYER_SIZE), 
#     activation = "relu", 
#     alpha=1e-4,
#     solver="sgd", 
#     learning_rate_init=.1
# )

# network.partial_fit(x, y, c)

# network.coefs_[0][:][:] = zeros(
#     shape=(INPUT_LAYER_SIZE,HIDDEN_LAYER_SIZE)
# )
# network.coefs_[1][:][:] = zeros(
#     shape=(HIDDEN_LAYER_SIZE,OUTPUT_LAYER_SIZE)
# )


# data = {
#     "weights":[],
#     "scores":[]
# }
# total = NeuralGridSearch.max_range(OUTPUT_LAYER_SIZE)
# try:
#     for ws1 in NeuralGridSearch.binary_vector_range(0,OUTPUT_LAYER_SIZE,10):
#         for ws2 in NeuralGridSearch.binary_vector_range(0,OUTPUT_LAYER_SIZE,10):

#             network.coefs_[1][:] = [ws1,ws2]
#             data["weights"].append(ws1+ws2)
#             data["scores"].append(network.score(x,y))
#             print(network.coefs_)
#             print()
# except Exception as e:
#     print(e)
# dump(data, open("test.pkl", 'wb'))

# data = read_pickle("test.pkl")
# print(data)
# projector = AutoEncoder(data["weights"], save_model=False )
# data["coordinates"] = projector.reduce_dimensions(data["weights"])
# dump(data, open("test.pkl", 'wb'))



# PLOTTING HEAT MAP OF PARAMETER SPACE
# data = read_pickle("test_10.pkl")
# x_min = min(coordinate[0] for coordinate in data["coordinates"])
# y_min = min(coordinate[1] for coordinate in data["coordinates"])
# x_max = max(coordinate[0] for coordinate in data["coordinates"])
# y_max = max(coordinate[1] for coordinate in data["coordinates"])
# print(x_min)
# print(y_min)
# print(x_max)
# print(y_max)

#100 skip
#matrix = zeros(shape=(23,23))
#def float_to_int(x:float) -> int:
#    return int((x + y_min)*100) + 5

# #10 skip
# matrix = zeros(shape=(572,572))
# def float_to_int(x:float) -> int:
#     return int((x + y_min)*10) + 8



# for xy,score in zip(data["coordinates"],data["scores"]):
#     x,y = xy    
#     i = float_to_int(x)
#     j = float_to_int(y)
#     matrix[i][j] = max(matrix[i][j],score)

# heatmap(data=matrix)#, annot=True, linewidth=.5)
# show()


# #ANALYSING BEST

# max_score = 0.1125
# for weights,score in zip(data["weights"],data["scores"]):
#     if score >= max_score:
#         ws = [weights[:10],weights[10:]]
#         heatmap(data=ws, annot=True, linewidth=.5)
#         print(score)
#         show()
#         input()
