############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from SimpSOM import somNet
from numpy import array
############   LOCAL IMPORTS   ###########################
from projection_models.autoencoder import AutoEncoder
from data_types import Vectors
##########################################################
    
class AE_SOM(AutoEncoder):
    """ NonLinear Dimensionality reduction via an Autoencoder (AE) and then SOM"""

    def __init__(self,training_vectors:Vectors) -> None:
        super().__init__(
            training_vectors=training_vectors,
            hidden_layer_sizes_encoder_only=[1024,512,256],
            projection_dimension=128
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