############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
from pandas import read_pickle
############   LOCAL IMPORTS   ###########################
from trainers.initialising_weights_trainer import InitialisingWeightsTrainer
from visualisers.main_experiment_visualiser import Visualiser
from visualisers.weight_space_visualiser import WeightSpaceVisualiser
from visualisers.initialisation_experiment_visualiser import InitialisationVisualiser
from projection_models.principal_component_analysis import PrincipalComponentAnalysis
from projection_models.autoencoder import AutoEncoder
from projection_models.binary_encoder import BinaryEncoder
from projection_models.self_organising_map import SelfOrganisingMap
##########################################################
parser = ArgumentParser()
parser.add_argument("--projection", type=str, choices=("PrincipalComponentAnalysis","SelfOrganisingMap","AutoEncoder","BinaryEncoder"), default="AutoEncoder", help="reduce dimension of network parameters for plotting")
parser.add_argument("--iterations", type=int, choices=range(1,100), default=10, help="training iterations")
parser.add_argument("--resolution", type=int, choices=range(10,200), default=30, help="dimension of heatmap")
parser.add_argument("--samples", type=int, choices=range(1,500), default=3, help="number of samples to plot")
parser.add_argument("--repetitions", type=int, choices=range(1,10), default=2, help="number of times to repeat same sample")
parser.add_argument("--initialisation", type=str, choices=("min","max"), default="max", help="initialisations at best (max) or worst (min) places in weight space")
args = parser.parse_args()

trainer = InitialisingWeightsTrainer(
    projector = eval(args.projection),
    projector_type = args.projection,
)
data = trainer.learn(
    training_iterations=args.iterations, 
    score_selector=eval(args.initialisation),
    number_of_samples = args.samples,
    repetition_of_sample = args.repetitions,
)
Visualiser.plot_coordinates(data)
InitialisationVisualiser.plot_scores(data)
loaded_data = read_pickle(f"../data/weight_space_experiment/sample_size_every_10.pkl")
coordinates = trainer.trained_projector.reduce_dimensions(
    vectors= loaded_data["weights"]
)
x_coordinates,y_coordinates = list(zip(*coordinates))
WeightSpaceVisualiser._visualise_weight_space(
    x_coordinates=list(x_coordinates) + data["x coordinate"].to_list(),
    y_coordinates=list(y_coordinates) + data["y coordinate"].to_list(),
    scores=loaded_data["scores"] + data["score"].to_list(),
    resolution=args.resolution
)
