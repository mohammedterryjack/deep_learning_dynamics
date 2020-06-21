############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from SimpSOM import somNet #https://github.com/fcomitani/SimpSOM
from numpy import array
############   LOCAL IMPORTS   ###########################
from data_types import Vectors
from projection_models.projection_method import ProjectionMethod
##########################################################
FILENAME = "SOM_20x20_wPCA"

class SelfOrganisingMap(ProjectionMethod):
    """nonlinear dimensionality reduction via Kohonen's Self-organising-map (SOM)"""
    def __init__(self, training_vectors:Vectors) -> None:
        try:
            self.model = self.load_model(filename=FILENAME)
        except FileNotFoundError:
            self.model = self.train_model(training_vectors)
            self.save_model(
                model=self.model,
                filename=FILENAME
            )
    
    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        return self.model.project(array=array(vectors), colnum=0, show=True)
    
    def train_model(self,vectors:Vectors) -> somNet:
        model = somNet(netHeight=20, netWidth=20, data=array(vectors), PCI=True)
        model.train(0.01, 10000)
        #TODO: speed up training
        return model
