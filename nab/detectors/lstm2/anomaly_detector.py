import numpy as np

from nupic.encoders import ScalarEncoder

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable


class AnomalyNet(chainer.Chain):

    def __init__(self, input_size, hidden_size=64):
        super(AnomalyNet, self).__init__(
            enc=L.Linear(input_size, hidden_size),
            lstm=L.LSTM(hidden_size, hidden_size),
            pred_enc1=L.Linear(input_size, hidden_size),
            pred_enc2=L.Linear(hidden_size, hidden_size),
            pred_enc3=L.Linear(hidden_size, hidden_size),
            fc1=L.Linear(hidden_size * 2, hidden_size),
            fc2=L.Linear(hidden_size, hidden_size),
            fc3=L.Linear(hidden_size, 1),
        )

    def __call__(self, x):
        h = F.tanh(self.enc(x))
        return self.lstm(h)

    def detect(self, x, h):
        h1_1 = F.tanh(self.pred_enc1(x))
        h1_2 = F.tanh(self.pred_enc2(h1_1))
        h1_3 = F.tanh(self.pred_enc3(h1_2))
        h2_1 = F.concat((h, h1_3), axis=1)
        h2_2 = F.tanh(self.fc1(h2_1))
        h2_3 = F.tanh(self.fc2(h2_2))
        return self.fc3(h2_3)


class LSTMAnomalyDetector(object):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        hidden_size = 32
        self.net = AnomalyNet(self.input_size, hidden_size)
        self.optimizer = chainer.optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.net)
        self.loss_func = F.sigmoid_cross_entropy

        self.input_max = kwargs['input_max']
        self.input_min = kwargs['input_min']
        input_margin = (self.input_max - self.input_min + 1e-6) * 0.2
        self.lower = self.input_min - input_margin
        self.upper = self.input_max + input_margin
        self.max_value = 0
        self.min_value = 0

        self.max_width = 50
        self.max_chain = 50
        self.unchain_interval = 10
        self.learning_anomaly_ratio = 0.5
        self.iteration = 0
        self.last_anomaly_iteration = -1000
        self.loss_history = []
        self.inputs = []
        self.scores = []
        self.losses = []

        self.xp = np

    def predict(self, inputs):
        xp = self.xp
        input_value = inputs['value']
        if self.iteration == 0:
            self.inputs = [[input_value]] * self.max_width
            self.max_value = input_value
            self.min_value = input_value
            self.iteration += 1
            return 0
        if input_value < self.min_value:
            self.min_value = input_value
        elif input_value > self.max_value:
            self.max_value = input_value
        x = Variable(xp.asarray(self.inputs, dtype=np.float32))
        h = self.net(x)

        self.inputs.insert(0, [input_value])
        self.inputs.pop()
        x_raw = xp.asarray(self.inputs, dtype=np.float32)
        lower = self.lower - self.min_value
        upper = self.upper - self.max_value
        a_raw = xp.random.uniform(lower, upper, size=(self.max_width, self.input_size)).astype(np.float32)
        a_raw = xp.where(a_raw < 0, a_raw + self.min_value, a_raw + self.max_value)
        std = xp.std(x_raw, axis=0, keepdims=True)
        n_raw = xp.random.randn(self.max_width, self.input_size) * std * 0.5
        rand = xp.random.random((self.max_width, 1))
        n_raw = xp.where(rand.repeat(self.input_size, axis=1) < 0.2, n_raw, 0)
        n_raw[0, :] = 0
        rand = xp.random.random((self.max_width, 1))
        t_raw = xp.where(rand < self.learning_anomaly_ratio, 1, 0)
        t_raw[0, :] = 0
        x = Variable(xp.where(t_raw.repeat(self.input_size, axis=1) > 0, a_raw, x_raw))
        t = Variable(t_raw)
        # updating parameter causes error when i is too small
        y, loss = self._update(x, h, t, self.iteration > 1)
        score = F.sigmoid(y[0]).data[0]
        self.losses.append(loss)
        if (self.iteration + 1) % self.unchain_interval == 0 and len(self.losses) > self.max_chain:
            self.losses[-self.max_chain].unchain_backward()
            self.losses = self.losses[-self.max_chain:]
        if self.iteration < self.last_anomaly_iteration + 200:
            score = float(xp.clip(score, 0, 0.89))
        elif score >= 0.9:
            score = float(score)
            self.last_anomaly_iteration = self.iteration
        self.iteration += 1
        return score

    def _update(self, x, h, t, update=True):
        y = self.net.detect(x, h)
        loss = self.loss_func(y, t)
        if update:
            self.net.cleargrads()
            loss.backward()
            self.optimizer.update()
        return y, loss
