############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
from trainers.initialising_weights_trainer import InitialisingWeightsTrainer
from visualisers.main_experiment_visualiser import Visualiser
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
from numpy import zeros

trainer = InitialisingWeightsTrainer(
    initialisation_vector = zeros(shape=(2,10)),
    projector = eval(args.projection)
)
data = trainer.learn(training_iterations=args.iterations)
Visualiser.plot_coordinates(data)