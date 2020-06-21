############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPRegressor
############   LOCAL IMPORTS   ###########################
from projection_method import ProjectionMethod
from data_types import Vectors
##########################################################
    
class AutoEncoder(ProjectionMethod):
    """ NonLinear Dimensionality reduction via an Autoencoder"""
    def __init__(self, training_vectors:Vectors) -> None:
        self.encoder = self.get_encoder_from_autoencoder(
            vectors=training_vectors,
            trained_encoder_decoder = self.train_autoencoder(training_vectors)
        )

    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        return self.encoder.predict(vectors)
    
    @staticmethod
    def train_autoencoder(vectors:Vectors) -> MLPRegressor:
        model = MLPRegressor(
            random_state=1, 
            activation="relu",
            hidden_layer_sizes = (128,32,8,2,8,32,128),
            verbose=True,
            max_iter=1000,
        ) 
        model.fit(vectors,vectors)
        return model
    
    @staticmethod
    def get_encoder_from_autoencoder(vectors:Vectors, trained_encoder_decoder:MLPRegressor) -> MLPRegressor:
        model = MLPRegressor(
            random_state=1, 
            activation="relu",
            hidden_layer_sizes = (128,32,8)
        ) 
        dummy_output_vectors = [[0.,0.]]*len(vectors)
        model.fit(vectors,dummy_output_vectors)
        model.coefs_ = trained_encoder_decoder.coefs_[:4]
        return model