############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPRegressor
from SimpSOM import somNet
from numpy import zeros, array
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from data_types import Vectors
##########################################################
    
class AE_SOM(ProjectionMethod):
    """ NonLinear Dimensionality reduction via an Autoencoder (AE) and then SOM"""
    #TODO: make this inherit from AutoEncoder class - to reduce repetition of code
    def __init__(self, training_vectors:Vectors) -> None:
        self.encoder = self.get_encoder_from_autoencoder(
            vectors=training_vectors,
            trained_encoder_decoder = self.train_autoencoder(training_vectors)
        )
        
    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        vectors_ae = self.encoder.predict(vectors)
        som = self.train_som(vectors_ae)
        return som.project(array=array(vectors_ae), colnum=0, show=True)

    @staticmethod
    def train_som(vectors:Vectors) -> somNet:
        model = somNet(netHeight=20, netWidth=20, data=array(vectors), PCI=True)
        model.train(0.01, 10000)
        return model
    
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
