import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment.experiment import Experiment
from domainadaptation.visualizer import Visualizer

from tqdm import trange

class CANExperiment(Experiment):
    '''https://arxiv.org/abs/1901.00976'''

    def __init__(self, config):
        super().__init__(config)

    def __call__(self):
        pass
