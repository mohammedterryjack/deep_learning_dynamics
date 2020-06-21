############   NATIVE IMPORTS  ###########################
from pickle import dump, load
from typing import Any
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
##########################################################
class ProjectionMethod:
    """Dimensionality Reduction"""
    def save_model(self,model:Any,filename:str):
        dump(model, open(f"data/{filename}.sav", 'wb'))

    def load_model(self,filename:str) -> Any:
        return load(open(f"data/{filename}.sav", 'rb'))