############   NATIVE IMPORTS  ###########################
from typing import Tuple, List, Union
from copy import deepcopy
from pickle import dump, load
############ INSTALLED IMPORTS ###########################
from pandas import read_pickle, DataFrame
from sklearn.neural_network import MLPClassifier
from numpy import array, zeros
from matplotlib.cm import inferno
from matplotlib.colors import Normalize
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from projection_models.autoencoder import AutoEncoder
from data_loader import DataLoader
from data_types import DataFrameNames, Vectors, Labels
##########################################################
colour_scaler = Normalize()

class InitialisingWeightsTrainer:
    """Compare the learning trajectories of neural networks beginning from specific weight initialisations """

    def __init__(self, projector:ProjectionMethod,projector_type:str) -> None:
        self.x,self.y,self.classes = DataLoader.load("mnist")
        self.trained_projector = self._initialise_projector(projector,projector_class=projector_type) 
    
    @staticmethod
    def _initialise_projector(projector:ProjectionMethod, projector_class:str) -> ProjectionMethod:
        if projector_class == "AutoEncoder":
            try:
                return load(open("../data/trained_models/autoencoder_trained_on_sample_size_every_10.sav", 'rb'))
            except:
                pass 
        if projector_class == "SelfOrganisingMap":
            try:
                return load(open("../data/trained_models/som_trained_on_sample_size_every_10.sav", 'rb'))
            except:
                pass 
        trained_projector = projector(
            training_vectors=read_pickle("../data/weight_space_experiment/sample_size_every_10.pkl")["weights"], 
            save_model=False
        )
        if projector_class == "AutoEncoder":
            dump(trained_projector, open("../data/trained_models/autoencoder_trained_on_sample_size_every_10.sav", 'wb'))
        if projector_class == "SelfOrganisingMap":
            dump(trained_projector, open("../data/trained_models/som_trained_on_sample_size_every_10.sav", 'wb'))
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
            learning_rate_init=.1,
            verbose=True
        )
        network.partial_fit(x, y, classes)
        network.coefs_[0][:][:] = zeros(
            shape=(input_layer_size,hidden_layer_size)
        )
        network.coefs_[1][:][:] = initialisation_vector
        return network

    @staticmethod
    def _get_initialisation_vectors(score_selector:callable, max_vectors:int) -> List[array]:
        data = read_pickle("../data/weight_space_experiment/sample_size_every_10.pkl")
        optimal_score = score_selector(score for score in data["scores"])
        initialisations = [
            array(weight).reshape(2,10) for weight,score in zip(
                data["weights"],
                data["scores"]
            ) if score == optimal_score
        ]
        return initialisations[:max_vectors]

    def learn(self, training_iterations:int, number_of_samples:int, repetition_of_sample:int) -> DataFrame:
        """ N learning iterations for M neural networks """
        
        learning_dynamics = []
        scores = []
        network_labels = []
        sample_labels = []
        iterations = []
        repetitions = []
        network_index = 0

        for sample_index,initialisation_vector in enumerate(
            self._get_initialisation_vectors(
                score_selector=max, 
                max_vectors=number_of_samples,
            ) + self._get_initialisation_vectors(
                score_selector=min, 
                max_vectors=number_of_samples,
            )
        ):
            for repetition in range(repetition_of_sample):
                learning_dynamics_, scores_, network_labels_, iterations_ = self._learn(
                    classes = self.classes,
                    training_inputs = self.x, 
                    training_outputs = self.y, 
                    iterations = training_iterations,
                    network =  self._initialise_neural_network_for_mnist_with_weights_initialised_to_vector(
                        input_layer_size=784,
                        hidden_layer_size=2,
                        output_layer_size=10,
                        x = self.x,
                        y = self.y,
                        classes = self.classes,
                        initialisation_vector = initialisation_vector,
                    ),
                )
                sample_labels_ = [sample_index for _ in range(training_iterations)]
                repetitions_ = [repetition for _ in range(training_iterations)]
                network_labels_ = list(map(lambda label:f"network_{network_index}:{label}", network_labels_))

                repetitions.extend(repetitions_)
                learning_dynamics.extend(learning_dynamics_)
                scores.extend(scores_)
                network_labels.extend(network_labels_)
                iterations.extend(iterations_)
                sample_labels.extend(sample_labels_)
                network_index += 1

        return self._wrap_as_dataframe(
            coordinates=self.trained_projector.reduce_dimensions(learning_dynamics),
            vectors=learning_dynamics,
            network_scores=scores,
            network_names=network_labels,
            sample_names= sample_labels,
            iteration_names=iterations,
            repetition_names = repetitions,
            initialisation_states = (
                ["max"]*training_iterations*repetition_of_sample*number_of_samples
            ) + (
                ["min"]*training_iterations*repetition_of_sample*number_of_samples
            )
        )

    @staticmethod
    def _wrap_as_dataframe(
        coordinates:Vectors,
        vectors:Vectors, 
        network_names:Labels, 
        network_scores:List[float],
        iteration_names:List[int],
        repetition_names:List[int],
        sample_names:Labels,
        initialisation_states:Labels
    ) -> DataFrame:
        colour_scaler.autoscale(network_scores)
        data = DataFrame(data=coordinates, columns=[DataFrameNames.X_COORDINATE,DataFrameNames.Y_COORDINATE])
        data[DataFrameNames.VECTOR] = vectors
        data[DataFrameNames.NETWORK_NAME] = network_names
        data[DataFrameNames.NETWORK_SCORE] = network_scores
        data[DataFrameNames.NETWORK_ITERATION] = iteration_names
        data[DataFrameNames.SAMPLE] = sample_names
        data[DataFrameNames.REPETITION] = repetition_names
        data[DataFrameNames.INITIALISATION_STATE] = initialisation_states
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
    ) -> Tuple[Vectors,List[float],Labels,Labels]:
        """ N learning iterations for a single neural network """

        vectors_over_time = []
        scores_over_time = []
        network_names = []
        iterations_per_network = []
        for iteration in range(iterations):
            vectors = InitialisingWeightsTrainer._step(
                network=network, 
                classes=classes, 
                training_inputs=training_inputs, 
                training_outputs=training_outputs, 
                skip_training = iteration==0,
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
                iterations_per_network.append(iteration)
        return vectors_over_time, scores_over_time, network_names, iterations_per_network


    @staticmethod
    def _step(
        network:MLPClassifier, 
        classes:Labels,
        training_inputs:Vectors, 
        training_outputs:Labels,
        skip_training:bool=False,
    ) -> Union[Vectors,array]: 
        """ single learning iteration for a single neural network """
        if not skip_training:
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
