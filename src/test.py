############   NATIVE IMPORTS  ###########################
from random import randint
from typing import List, Iterable
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPClassifier
from numpy import ones,array
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
from neural_grid_search import NeuralGridSearch
##########################################################

#TODO: heatmap of single hidden layers weight space. Get accuracy for each coordinate (find corresponding weight for that coordinate and test network)
#TODO: if initialised in good areas of the weight space - and see how this affects training

INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 10
first_weights = ones(
    shape=(INPUT_LAYER_SIZE,HIDDEN_LAYER_SIZE)
)
second_weights = ones(
    shape=(HIDDEN_LAYER_SIZE,OUTPUT_LAYER_SIZE)
)
weights = [
    first_weights,
    second_weights
]

x,y,c = DataLoader.load("mnist")

network = MLPClassifier(
    hidden_layer_sizes= (HIDDEN_LAYER_SIZE), 
    activation = "relu", 
    alpha=1e-4,
    solver="sgd", 
    learning_rate_init=.1
)

network.partial_fit(x, y, c)

# for i in range(0,HIDDEN_LAYER_SIZE):
#     #network.coefs_ = weights
#     for j in range(0,OUTPUT_LAYER_SIZE):
#         for w in weight_range(step_size=.5):
#             network.coefs_[1][i][j] = w
#             print(network.coefs_)
#             print(network.score(x,y))

t = NeuralGridSearch.max_range(OUTPUT_LAYER_SIZE)*2
network.coefs_[0][:][:] = first_weights
for i,ws1 in enumerate(NeuralGridSearch.greedy_search(OUTPUT_LAYER_SIZE)):
    for j,ws2 in enumerate(NeuralGridSearch.greedy_search(OUTPUT_LAYER_SIZE)):
        network.coefs_[1][0][:] = ws1
        network.coefs_[1][1][:] = ws2
        print("progress=",(i+j)/t)
        #print(network.coefs_)
        print("score=",network.score(x,y))
        print()
