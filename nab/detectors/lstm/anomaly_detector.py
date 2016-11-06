import numpy as np
import six

from nupic.encoders import ScalarEncoder

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import Variable


class AnomalyNet(chainer.Chain):

    def __init__(self, input_size, hidden_size=64):
        super(AnomalyNet, self).__init__(
            enc=L.Linear(input_size, hidden_size),
            lstm=L.StatelessLSTM(hidden_size, hidden_size),
            dec=L.Linear(hidden_size, input_size),
        )

    def __call__(self, x, state):
        c, h = state
        h1 = F.tanh(self.enc(x))
        c_next, h_next = self.lstm(c, h, h1)
        y = self.dec(h_next)
        return y, (c_next, h_next)

    def initial_state(self):
        return (None, None)

class LSTMAnomalyDetector(object):

    def __init__(self, *args, **kwargs):
        input_size = 32
        hidden_size = 16
        bit_length = 7
        self.net = AnomalyNet(input_size, hidden_size)
        self.optimizer = chainer.optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.net)
        self.loss_func = F.sigmoid_cross_entropy

        input_max = kwargs['input_max']
        input_min = kwargs['input_min']
        rangePadding = abs(input_max - input_min) * 0.2 + 1e-4
        self.encoder = ScalarEncoder(bit_length, input_min - rangePadding, input_max + rangePadding, n=input_size, forced=True)

        self.max_width = 30
        self.max_chain = 50
        self.skip_width = 3
        self.skip_step = 10
        self.unchain_interval = 10
        self.iteration = 0
        self.last_anomaly_iteration = -1000
        self.long_term = 1000
        self.short_term = 10
        self.loss_history = []
        self.inputs = []
        self.scores = []
        self.losses = []
        self.states = [self.net.initial_state()] * self.skip_width

        self.xp = np

    def predict(self, inputs):
        input_value = self._encode(inputs['value'])
        if self.iteration == 0:
            self.inputs = [input_value] * (self.max_width * self.skip_step)
            self.iteration += 1
            return 0
        x = self.xp.asarray(self.inputs[::self.skip_step], dtype=np.float32)
        self.inputs.insert(0, input_value)
        self.inputs.pop()
        t = self.xp.asarray(self.inputs[::self.skip_step], dtype=np.int32)
        t0 = cuda.to_cpu(t[0])
        losses = []
        for i in six.moves.range(self.skip_width):
            y, state, loss = self._update(x[i::self.skip_width], t[i::self.skip_width], self.states[i], self.iteration > 1)
            self.states[i] = state
            losses.append(loss)
            if i == 0:
                y0 = cuda.to_cpu(F.sigmoid(y).data[0])
        score = 1 - float(np.dot(y0, t0) / np.sum(t0))
        self.scores.append(score)
        if len(self.scores) > self.long_term:
            self.scores = self.scores[-self.long_term:]
        mean = float(np.mean(self.scores))
        std = float(np.std(self.scores)) + 1e-10
        recent_mean = float(np.mean(self.scores[-self.short_term:]))
        self.losses.append(losses)
        if (self.iteration + 1) % self.unchain_interval == 0 and len(self.losses) > self.max_chain:
            for loss in self.losses[-self.max_chain]:
                loss.unchain_backward()
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

    def _update(self, x, t, state, update=True):
        y, new_state = self.net(x, state)
        loss = self.loss_func(y, t)
        if update:
            self.net.cleargrads()
            loss.backward()
            self.optimizer.update()
        return y, new_state, loss
