############   NATIVE IMPORTS  ###########################
from pickle import dump, load
from typing import Any
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
##########################################################
class ProjectionMethod:
    """Dimensionality Reduction"""
    def save_model(self,model:Any,filename:str):
        dump(model, open(f"../data/trained_models/{filename}.sav", 'wb'))

    def load_model(self,filename:str) -> Any:
        return load(open(f"../data/trained_models/{filename}.sav", 'rb'))