############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
from numpy import zeros
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from data_types import Vectors
##########################################################
    
class AE_Tsne(ProjectionMethod):
    """ NonLinear Dimensionality reduction via an Autoencoder (AE) and then Tsne"""
    #TODO: make this inherit from AutoEncoder class - to reduce repetition of code

    def __init__(self, training_vectors:Vectors) -> None:
        self.encoder = self.get_encoder_from_autoencoder(
            vectors=training_vectors,
            trained_encoder_decoder = self.train_autoencoder(training_vectors)
        )

    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        return TSNE(n_components=2).fit_transform(
            self.encoder.predict(vectors)
        )
    
    @staticmethod
    def train_autoencoder(vectors:Vectors) -> MLPRegressor:
        model = MLPRegressor(
            random_state=1, 
            activation="relu",
            hidden_layer_sizes = (1024,512,256,128,256,512,1024),
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
            hidden_layer_sizes = (1024,512,256)
        ) 
        OUTPUT_LAYER_SIZE = 128
        dummy_output_vectors = zeros(shape=(len(vectors),OUTPUT_LAYER_SIZE))
        model.fit(vectors,dummy_output_vectors)
        model.coefs_ = trained_encoder_decoder.coefs_[:4]
        return model
