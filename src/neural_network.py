############   NATIVE IMPORTS  ###########################
from typing import Optional, Tuple, List, Union, Union
from datetime import datetime 
from json import dumps
from copy import deepcopy
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPClassifier
from numpy import array, concatenate, mean
from matplotlib.cm import inferno
from matplotlib.colors import Normalize
from pandas import DataFrame
############   LOCAL IMPORTS   ###########################
from data_types import Vectors,Labels,DataFrameNames
from projection_models.projection_method import ProjectionMethod
##########################################################
TRAINING_MESSAGE = """
network = {network_index}
iteration = {training_iteration}
score = {score}
layers of shape {layer_shapes} were converted into a 1d vector of shape {vector_shape}
using {layer_combination_method}
"""
colour_scaler = Normalize()

class DeepNeuralNetworkTrainer:
    """Simple Multi-layered Feed-forward Neural Networks"""

    @staticmethod
    def learn(
        training_inputs:Vectors, 
        training_outputs:Labels, 
        classes:Labels,
        training_iterations:int, 
        projection_model:ProjectionMethod,
        train_projection_model_on_first_only:bool=False,
        number_of_networks:int=1, 
        network_parameters:Optional[dict]=None,
        layer_to_track:Optional[int]=None, 
        average_hidden_layers:bool=False, 
        notes:str="",
        track_first_n_layers_separately:bool=False
    ) -> DataFrame:
        """ N learning iterations for M neural networks """
                
        vectorisation_method = DeepNeuralNetworkTrainer._get_vectorisation_method(
            layer_to_track=layer_to_track, 
            average_hidden_layers=average_hidden_layers,
            enough_layers=network_parameters and "hidden_layer_sizes" in network_parameters and len(network_parameters["hidden_layer_sizes"]) > 2,
            track_first_n_layers_separately=track_first_n_layers_separately
        )
        learning_history = []
        network_names = []
        network_scores = []
        for network_index in range(number_of_networks):
            network = MLPClassifier(**network_parameters) if network_parameters else MLPClassifier()
            learning_dynamics, scores, labels = DeepNeuralNetworkTrainer._learn(
                network = network,
                network_id = network_index,
                classes = classes,
                training_inputs = training_inputs, 
                training_outputs = training_outputs, 
                iterations = training_iterations,
                vectorisation_method = vectorisation_method,
                layers_to_track = track_first_n_layers_separately if track_first_n_layers_separately else 1,
            )
            learning_history.extend(learning_dynamics)
            network_scores.extend(scores)
            network_names.extend(labels)
        
        trained_projection_model = projection_model(
            training_vectors=learning_history[:training_iterations] if train_projection_model_on_first_only else learning_history
        )
        dataframe = DeepNeuralNetworkTrainer.wrap_as_dataframe(
            coordinates=trained_projection_model.reduce_dimensions(learning_history),
            vectors=learning_history,
            network_scores=network_scores,
            network_names=network_names
        )
        DeepNeuralNetworkTrainer.save(
            data=dataframe,
            meta_data={
                "notes":notes,
                "training_iterations":training_iterations, 
                "projection_model":type(trained_projection_model).__name__,
                "train_projection_model_on_first_only":train_projection_model_on_first_only,
                "number_of_networks":number_of_networks, 
                "network_parameters":network_parameters,
                "layer_to_track":layer_to_track, 
                "average_hidden_layers":average_hidden_layers, 
                "track_first_n_layers_separately":track_first_n_layers_separately,
            }
        )
        return dataframe


    @staticmethod
    def save(data:DataFrame,meta_data:dict) -> None:
        filename = DeepNeuralNetworkTrainer.now_as_a_string()
        data.to_pickle(f"../data/{filename}.pkl")
        with open(f"../data/{filename}.json",'w') as file_to_write_to:
            file_to_write_to.write(dumps(meta_data,indent=3))
        print(data)


    @staticmethod
    def _learn(
        network:MLPClassifier, 
        network_id:int,
        classes:Labels,
        training_inputs:Vectors, 
        training_outputs:Labels, 
        iterations:int,
        vectorisation_method:Optional[callable],
        layers_to_track:int
    ) -> Tuple[Vectors,List[float],Labels]:
        """ N learning iterations for a single neural network """

        vectors_over_time = []
        scores_over_time = []
        network_names = []
        for i in range(iterations):
            vectors = DeepNeuralNetworkTrainer._step(
                network=network, 
                classes=classes, 
                training_inputs=training_inputs, 
                training_outputs=training_outputs, 
                convert_matrix_to_vector_method= vectorisation_method 
            )                
            score = network.score(training_inputs, training_outputs)
            if vectorisation_method:
                vectors_over_time.append(vectors)
                scores_over_time.append(score)
                network_names.append(f"network:{network_id}")
            else:
                for layer_id in range(1,layers_to_track):
                    vectors_over_time.append(
                        DeepNeuralNetworkTrainer._convert_matrix_into_vector_by_sampling(
                            vectors=vectors,
                            layer_to_sample=layer_id
                        )
                    )
                    scores_over_time.append(score)
                    network_names.append(f"network:{network_id}_layer:{layer_id}")

            print(TRAINING_MESSAGE.format(
                network_index = network_id,
                training_iteration = i,
                layer_shapes = list(map(lambda layer:layer.shape, network.coefs_)),
                vector_shape = vectors_over_time[-1].shape,
                score = scores_over_time[-1],
                layer_combination_method = vectorisation_method
            ))

        return vectors_over_time, scores_over_time, network_names


    @staticmethod
    def _step(
        network:MLPClassifier, 
        classes:Labels,
        training_inputs:Vectors, 
        training_outputs:Labels,
        convert_matrix_to_vector_method:Optional[callable],
    ) -> Union[Vectors,array]: 
        """ single learning iteration for a single neural network """
        network.partial_fit(training_inputs, training_outputs, classes)
        network_layers_as_vectors = list(map(DeepNeuralNetworkTrainer._convert_matrix_into_vector_by_reshaping,network.coefs_))
        return convert_matrix_to_vector_method(network_layers_as_vectors) if convert_matrix_to_vector_method else network_layers_as_vectors
    

    @staticmethod
    def _get_vectorisation_method(
        layer_to_track:Optional[int], 
        average_hidden_layers:bool,
        enough_layers:bool,
        track_first_n_layers_separately:int,
    ) -> Optional[callable]:
        """ method to convert N 1d vectors (neural network layers) into a single 1d vector """

        if track_first_n_layers_separately:
            return
        if layer_to_track is not None:
            return lambda layers: DeepNeuralNetworkTrainer._convert_matrix_into_vector_by_sampling(
                vectors=layers,
                layer_to_sample=layer_to_track
            )
        if average_hidden_layers and enough_layers:
            return DeepNeuralNetworkTrainer._average_hidden_layers_and_concatenate_input_output_layers 
        return concatenate

    @staticmethod
    def _average_hidden_layers_and_concatenate_input_output_layers(vectors:Vectors) -> array:
        return concatenate(
            [
                vectors[0],
                DeepNeuralNetworkTrainer._convert_matrix_into_vector_by_averaging(
                    vectors = vectors[1:-2]
                ),
                vectors[-1],
            ]
        ) 

        
    @staticmethod
    def _convert_matrix_into_vector_by_sampling(vectors:Vectors, layer_to_sample:int) -> array:
        """ layer to sample = 0: [[1,2],[3,4]] -> [1,2] """
        return deepcopy(vectors[layer_to_sample])


    @staticmethod
    def _convert_matrix_into_vector_by_averaging(vectors:Vectors) -> array:
        """ [[1,2],[3,4]] -> [2, 3] """
        return mean(vectors,axis=0)


    @staticmethod
    def _convert_matrix_into_vector_by_reshaping(nested_array:array) -> array:
        """ [[1,2],[3,4]] -> [1,2,3,4]"""
        return nested_array.reshape(-1)

    @staticmethod
    def wrap_as_dataframe(coordinates:Vectors, vectors:Vectors, network_names:Labels, network_scores:List[float]) -> DataFrame:
        colour_scaler.autoscale(network_scores)
        data = DataFrame(data=coordinates, columns=[DataFrameNames.X_COORDINATE,DataFrameNames.Y_COORDINATE])
        data[DataFrameNames.VECTOR] = vectors
        data[DataFrameNames.NETWORK_NAME] = network_names
        data[DataFrameNames.NETWORK_SCORE] = network_scores
        data[DataFrameNames.COLOUR] = list(map(list, inferno(colour_scaler(network_scores))))
        return data        

    @staticmethod
    def now_as_a_string() -> str:
        """ returns a nicely formatted string of the current time"""
        return datetime.now().strftime("%Hh%Mm%Ss%d%B%Y")
