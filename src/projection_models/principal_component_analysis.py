############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.decomposition import PCA
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from data_types import Vectors
##########################################################

class PrincipalComponentAnalysis(ProjectionMethod):
    """ Linear Dimensionality reduction via Principal Component Analysis (PCA)"""
    def __init__(self, training_vectors:Vectors) -> None:
        self.projection_method = PCA(n_components=2)
        self.projection_method.fit(training_vectors)

    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        """ [[1,2,3,4,5],[1,2,3,4,5]] -> [[.5,.5], [.5,.5]] """
        return self.projection_method.transform(vectors)
