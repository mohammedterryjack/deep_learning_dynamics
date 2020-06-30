############   NATIVE IMPORTS  ###########################
from typing import Tuple, List, Union
from copy import deepcopy
from pickle import dump, load
############ INSTALLED IMPORTS ###########################
from pandas import read_pickle, DataFrame
from sklearn.neural_network import MLPClassifier
from numpy import zeros, array
from matplotlib.cm import inferno
from matplotlib.colors import Normalize
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from projection_models.autoencoder import AutoEncoder
from data_loader import DataLoader
from data_types import dataDataFrameNames, Vectors, Labels
##########################################################
colour_scaler = Normalize()

class InitialisingWeightsTrainer:
    """Compare the learning trajectories of neural networks beginning from specific weight initialisations """

    def __init__(self, initialisation_vector:array, projector:ProjectionMethod) -> None:
        self.x,self.y,self.classes = DataLoader.load("mnist")
        self.network = self._initialise_neural_network_for_mnist_with_weights_initialised_to_vector(
            input_layer_size=784,
            hidden_layer_size=2,
            output_layer_size=10,
            x = self.x,
            y = self.y,
            classes = self.classes,
            initialisation_vector = initialisation_vector,
        )
        self.trained_projector = self._initialise_projector(projector) 
    
    @staticmethod
    def _initialise_projector(projector:ProjectionMethod) -> ProjectionMethod:
        if isinstance(projector, AutoEncoder):
            try:
                return load(open("../data/trained_models/autoencoder_trained_on_sample_size_every_10.sav", 'rb'))
            except:
                pass 
        training_vectors = read_pickle("../data/weight_space_experiment/sample_size_every_10.pkl")["weights"]
        trained_projector = projector(
            training_vectors=training_vectors, 
            save_model=False
        )
        if isinstance(trained_projector, AutoEncoder):
            dump(trained_projector, open("../data/trained_models/autoencoder_trained_on_sample_size_every_10.sav", 'wb'))
        return trained_projector

    @staticmethod
    def _initialise_neural_network_for_mnist_with_weights_initialised_to_vector(
        input_layer_size:int, 
        hidden_layer_size:int, 
        output_layer_size:10,
        x:Vectors,
        y:Labels,
        classes:Labels,
        initialisation_vector:array
    ) -> MLPClassifier:
        assert(initialisation_vector.shape == (hidden_layer_size,output_layer_size))
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
        network.coefs_[1][:][:] = initialisation_vector
        return network

    def learn(self, training_iterations:int) -> DataFrame:
        """ N learning iterations for M neural networks """
                
        learning_dynamics, scores, labels = self._learn(
            network = self.network,
            classes = self.classes,
            training_inputs = self.x, 
            training_outputs = self.y, 
            iterations = training_iterations,
        )
        return self._wrap_as_dataframe(
            coordinates=self.trained_projector.reduce_dimensions(learning_dynamics),
            vectors=learning_dynamics,
            network_scores=scores,
            network_names=labels
        )

    @staticmethod
    def _wrap_as_dataframe(coordinates:Vectors, vectors:Vectors, network_names:Labels, network_scores:List[float]) -> DataFrame:
        colour_scaler.autoscale(network_scores)
        data = DataFrame(data=coordinates, columns=[DataFrameNames.X_COORDINATE,DataFrameNames.Y_COORDINATE])
        data[DataFrameNames.VECTOR] = vectors
        data[DataFrameNames.NETWORK_NAME] = network_names
        data[DataFrameNames.NETWORK_SCORE] = network_scores
        data[DataFrameNames.COLOUR] = list(map(list, inferno(colour_scaler(network_scores))))
        print(data)
        return data        

    @staticmethod
    def _learn(
        network:MLPClassifier, 
        classes:Labels,
        training_inputs:Vectors, 
        training_outputs:Labels, 
        iterations:int,
    ) -> Tuple[Vectors,List[float],Labels]:
        """ N learning iterations for a single neural network """

        vectors_over_time = []
        scores_over_time = []
        network_names = []
        for i in range(iterations):
            vectors = InitialisingWeightsTrainer._step(
                network=network, 
                classes=classes, 
                training_inputs=training_inputs, 
                training_outputs=training_outputs, 
            )                
            score = network.score(training_inputs, training_outputs)
            for layer_id in range(1,2):
                vectors_over_time.append(
                    InitialisingWeightsTrainer._convert_matrix_into_vector_by_sampling(
                        vectors=vectors,
                        layer_to_sample=layer_id
                    )
                )
                scores_over_time.append(score)
                network_names.append(f"layer:{layer_id}")
        return vectors_over_time, scores_over_time, network_names


    @staticmethod
    def _step(
        network:MLPClassifier, 
        classes:Labels,
        training_inputs:Vectors, 
        training_outputs:Labels,
    ) -> Union[Vectors,array]: 
        """ single learning iteration for a single neural network """
        network.partial_fit(training_inputs, training_outputs, classes)
        return list(map(InitialisingWeightsTrainer._convert_matrix_into_vector_by_reshaping,network.coefs_))

    @staticmethod
    def _convert_matrix_into_vector_by_sampling(vectors:Vectors, layer_to_sample:int) -> array:
        """ layer to sample = 0: [[1,2],[3,4]] -> [1,2] """
        return deepcopy(vectors[layer_to_sample])

    @staticmethod
    def _convert_matrix_into_vector_by_reshaping(nested_array:array) -> array:
        """ [[1,2],[3,4]] -> [1,2,3,4]"""
        return nested_array.reshape(-1)
