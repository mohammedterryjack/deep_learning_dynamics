############   NATIVE IMPORTS  ###########################
from random import randint
from typing import List, Iterable
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPClassifier
from numpy import ones,array
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
##########################################################

#TODO: heatmap of single hidden layers weight space. Get accuracy for each coordinate (find corresponding weight for that coordinate and test network)
#TODO: if initialised in good areas of the weight space - and see how this affects training


def weight_range(min_value:int=-1, max_value:int=1,step_size:float=.01) -> float:
    for weight in range(
        int(min_value//step_size),
        int(max_value//step_size) + 1,
        1
    ):
        yield weight*step_size

def convert_binary_string_to_int(binary_string:str) -> int:
    return int(binary_string, 2)

def binary_string(number:int,leading_zeros:int) -> str:
    return format(number,f"#0{leading_zeros+2}b")[2:]

def binary_vector(number:int, vector_size:int) -> List[int]:
    return list(
        map(int,binary_string(number=number,leading_zeros=vector_size))
    )  

def weight_grid_search(vector_size:int) -> Iterable[List[int]]:
    max_iteration = convert_binary_string_to_int(
        binary_string='1'*vector_size
    )
    return list(
        map(
            lambda i:binary_vector(number=i,vector_size=vector_size),
            range(max_iteration+1)
        )
    )    

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

network.coefs_[0][:][:] = first_weights
for ws1 in weight_grid_search(OUTPUT_LAYER_SIZE):
    for ws2 in weight_grid_search(OUTPUT_LAYER_SIZE):
        network.coefs_[1][0][:] = ws1
        network.coefs_[1][1][:] = ws2
        print(network.coefs_)
        print(network.score(x,y))
