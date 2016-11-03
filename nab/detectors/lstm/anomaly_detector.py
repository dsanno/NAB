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
            dec=L.Linear(hidden_size, input_size),
        )

    def __call__(self, x):
        h = F.tanh(self.enc(x))
        h = self.lstm(h)
        y = self.dec(h)
        return y

class LSTMAnomalyDetector(object):

    def __init__(self, *args, **kwargs):
        input_size = 32
        hidden_size = 16
        bit_length = 7
        self.net = AnomalyNet(input_size, hidden_size)
        self.optimizer = chainer.optimizers.Adam(alpha=0.01)
        self.optimizer.setup(self.net)
        self.loss_func = F.sigmoid_cross_entropy

        input_max = kwargs['input_max']
        input_min = kwargs['input_min']
        rangePadding = abs(input_max - input_min) * 0.2 + 1e-4
        self.encoder = ScalarEncoder(bit_length, input_min - rangePadding, input_max + rangePadding, n=input_size, forced=True)

        self.max_width = 50
        self.max_chain = 20
        self.unchain_interval = 10
        self.iteration = 0
        self.last_anomaly_iteration = -1000
        self.long_term = 1000
        self.short_term = 10
        self.loss_history = []
        self.inputs = []
        self.scores = []
        self.losses = []

        self.xp = np

    def predict(self, inputs):
        input_value = self._encode(inputs['value'])
        if self.iteration == 0:
            self.inputs = [input_value] * self.max_width
            self.iteration += 1
            return 0
        x = Variable(self.xp.asarray(self.inputs, dtype=np.float32))
        self.inputs.insert(0, input_value)
        self.inputs.pop()
        t = Variable(self.xp.asarray(self.inputs, dtype=np.int32))
        y, loss = self._update(x, t, self.iteration > 1)
        y_raw = F.sigmoid(y).data[0]
        t_raw = t.data[0]
        self.scores.append(1 - float(np.dot(y_raw, t_raw) / np.sum(t_raw)))
        if len(self.scores) > self.long_term:
            self.scores = self.scores[-self.long_term:]
        mean = float(np.mean(self.scores))
        std = float(np.std(self.scores)) + 1e-10
        recent_mean = float(np.mean(self.scores[-self.short_term:]))
        self.losses.append(loss)
        if (self.iteration + 1) % self.unchain_interval == 0 and len(self.losses) > self.max_chain:
            self.losses[-self.max_chain].unchain_backward()
            self.losses = self.losses[-self.max_chain:]
        if self.iteration < self.last_anomaly_iteration + 200:
            score = float(np.clip((recent_mean - mean) / std * 0.5, 0, 0.50))
        else:
            score = float(np.clip((recent_mean - mean) / std * 0.5, 0, 1))
            if score >= 0.7:
                self.last_anomaly_iteration = self.iteration
        self.iteration += 1
        return score

    def _encode(self, value):
        x = np.zeros((self.encoder.getWidth(),), dtype=np.int32)
        self.encoder.encodeIntoArray(value, x)
        return x.tolist()

    def _update(self, x, t, update=True):
        y = self.net(x)
        loss = self.loss_func(y, t)
        if update:
            self.net.cleargrads()
            loss.backward()
            self.optimizer.update()
        return y, loss
