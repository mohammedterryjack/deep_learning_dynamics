############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
from trainers.weight_space_trainer import WeightSpaceTrainer
from visualisers.weight_space_visualiser import WeightSpaceVisualiser
from projection_models.principal_component_analysis import PrincipalComponentAnalysis
from projection_models.autoencoder import AutoEncoder
from projection_models.binary_encoder import BinaryEncoder
from projection_models.self_organising_map import SelfOrganisingMap
##########################################################
parser = ArgumentParser()
parser.add_argument("--ignore_first_layer", action='store_true', default=False, help="if ignored, the first layer will remain initialised with random weights")
parser.add_argument("--search_step_size", type=int, default=100, help="to search through entire weight space make step size = 1")
parser.add_argument("--projection", type=str, choices=("PrincipalComponentAnalysis","SelfOrganisingMap","AutoEncoder","BinaryEncoder"), default="AutoEncoder", help="reduce dimension of network parameters for plotting")
parser.add_argument("--filename", type=str, help="load in a previous datafile using its filenmame")
args = parser.parse_args()

filename = args.filename
if not filename:
    filename = WeightSpaceTrainer().map_weight_space(
        sample_step_size=args.search_step_size,
        ignore_first_layer=args.ignore_first_layer
    )
WeightSpaceVisualiser(
    projector = eval(args.projection),
    data_filename=filename
).visualise_weight_space()