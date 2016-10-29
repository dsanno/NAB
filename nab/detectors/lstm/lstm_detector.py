import numpy as np
import random
from nab.detectors.base import AnomalyDetector

import chainer
from chainer import functions as F
from chainer import optimizers

from anomaly_detector import LSTMAnomalyDetector


class LstmDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(LstmDetector, self).__init__(*args, **kwargs)

        self.seed = 1

    def initialize(self):
        np.random.seed(self.seed)
        self.detector = LSTMAnomalyDetector(
            input_max=self.inputMax,
            input_min=self.inputMin
        )

    def handleRecord(self, inputData):
        """Returns a tuple (anomalyScore).
        The anomalyScore is simply a random value from 0 to 1
        """
        anomalyScore = self.detector.predict(inputData)
        return (anomalyScore, )
