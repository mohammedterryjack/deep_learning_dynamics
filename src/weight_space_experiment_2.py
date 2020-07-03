############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
from pandas import read_pickle
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
parser.add_argument("--initialistion", type=str, choices=("min","max"), default="max", help="initialisations at best or worst places in weight space")
args = parser.parse_args()

#TODO: initialise from several good weight spaces and plot learning
#TODO: initialise from several bad weight spaces and plot learning
#TODO: plot training of both on same plot (average)
#TODO: train projectors on data from other experiment as it has more range of values 
#TODO: if initialised in good areas of the weight space (determined by other experiment) - and see how this affects training

trainer = InitialisingWeightsTrainer(
    projector = eval(args.projection),
    projector_type = args.projection 
)
data = trainer.learn(training_iterations=args.iterations, score_selector=eval(args.initialistion))
Visualiser.plot_coordinates(data)
loaded_data = read_pickle(f"../data/weight_space_experiment/sample_size_every_10.pkl")
coordinates = trainer.trained_projector.reduce_dimensions(
    vectors= loaded_data["weights"]
)
x_coordinates,y_coordinates = list(zip(*coordinates))
WeightSpaceVisualiser._visualise_weight_space(
    x_coordinates=list(x_coordinates) + data["x coordinate"].to_list(),
    y_coordinates=list(y_coordinates) + data["y coordinate"].to_list(),
    scores=loaded_data["scores"] + data["score"].to_list(),
    resolution=30
)
