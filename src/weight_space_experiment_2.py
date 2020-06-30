############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
from pandas import merge
############   LOCAL IMPORTS   ###########################
from trainers.initialising_weights_trainer import InitialisingWeightsTrainer
from visualisers.main_experiment_visualiser import Visualiser
from visualisers.weight_space_visualiser import WeightSpaceVisualiser
from projection_models.principal_component_analysis import PrincipalComponentAnalysis
from projection_models.autoencoder import AutoEncoder
from projection_models.binary_encoder import BinaryEncoder
from projection_models.self_organising_map import SelfOrganisingMap
##########################################################
parser = ArgumentParser()
parser.add_argument("--projection", type=str, choices=("PrincipalComponentAnalysis","SelfOrganisingMap","AutoEncoder","BinaryEncoder"), default="AutoEncoder", help="reduce dimension of network parameters for plotting")
parser.add_argument("--iterations", type=int, choices=range(1,100), default=10, help="training iterations")
args = parser.parse_args()

#TODO: initialise from good weight spaces and plot learning
#TODO: initialise from bad weight spaces and plot learning
#TODO: plot training of both on same plot (average)
#TODO: train projectors on data from other experiment as it has more range of values 
#TODO: if initialised in good areas of the weight space (determined by other experiment) - and see how this affects training
from numpy import ones

trainer = InitialisingWeightsTrainer(
    initialisation_vector = ones(shape=(2,10)),
    projector = eval(args.projection),
    autoencoder_selected = args.projection == "AutoEncoder"
)
data = trainer.learn(training_iterations=args.iterations)
Visualiser.plot_coordinates(data)
trainer.trained_projector
print(wsv.data)
print()
print(data)
#TODO: merge dataframes so that heatmap shows brighter patch of the learning trajectory
#data: ["x coordinate", "y coordinate", "score"] ["color"]
#merged_data = merge(wsv.data,data[['Key_Column','Target_Column']],on='Key_Column', how='left')
#wsv.data = merged_data 
WeightSpaceVisualiser._visualise_weight_space(
    coordinates=,
    scores=,
    resolution=30
)
