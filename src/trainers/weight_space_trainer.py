############   NATIVE IMPORTS  ###########################
from typing import List,Optional
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPClassifier
from numpy import zeros
from pickle import dump
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
from neural_grid_search import NeuralGridSearch
from data_types import Vectors, Labels
##########################################################

class WeightSpaceTrainer:
    """Map the accuracy of a given neural network's in weightspace """

    def __init__(self) -> None:
        self.x,self.y,classes = DataLoader.load("mnist")
        self.network = self._initialise_neural_network_for_mnist_with_weights_zeroed(
            input_layer_size=784,
            hidden_layer_size=2,
            output_layer_size=10,
            x = self.x,
            y = self.y,
            classes = classes
        )
    
    def map_weight_space(self,data_filename:Optional[str]=None, sample_step_size:int=10) -> str:
        scores = self._get_scores_for_weight_values(
            output_layer_size=10,
            sample_step_size=sample_step_size,
            network=self.network,
            x = self.x,
            y = self.y
        )
        data_filename = data_filename if data_filename else f"sample_size_every_{sample_step_size}"
        dump(scores, open(f"../../data/weight_space_experiment/{data_filename}.pkl", 'wb'))
        return data_filename

   @staticmethod
    def _initialise_neural_network_for_mnist_with_weights_zeroed(
        input_layer_size:int, 
        hidden_layer_size:int, 
        output_layer_size:10,
        x:Vectors,
        y:Labels,
        classes:Labels
    ) -> MLPClassifier:
        network = MLPClassifier(
            hidden_layer_sizes= (hidden_layer_size), 
            activation = "relu", 
            alpha=1e-4,
            solver="sgd", 
            learning_rate_init=.1
        )
        network.partial_fit(x, y, classes)
        network.coefs_[0][:][:] = zeros(
            shape=(input_layer_size,hidden_layer_size)
        )
        network.coefs_[1][:][:] = zeros(
            shape=(hidden_layer_size,output_layer_size)
        )
        return network

    @staticmethod
    def _get_scores_for_weight_values(
        output_layer_size:int,
        sample_step_size:int,
        network:MLPClassifier,
        x:Vectors,
        y:Labels
    ) -> Dict[Vectors,List[float]]:
        data = {
            "weights":[],
            "scores":[]
        }
        for weights_1 in NeuralGridSearch.binary_vector_range(0,output_layer_size,sample_step_size):
            for weights_2 in NeuralGridSearch.binary_vector_range(0,output_layer_size,sample_step_size):
                network.coefs_[1][:] = [
                    weights_1,
                    weights_2
                ]
                data["weights"].append(weights_1+weights_2)
                data["scores"].append(network.score(x,y))
        return data 