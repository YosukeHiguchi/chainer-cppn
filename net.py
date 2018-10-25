import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, n_unit, n_depth, cdim, mode):
        self.n_unit = n_unit
        self.n_depth = n_depth # >= 2
        self.mode = mode

        w = chainer.initializers.Normal(scale=1.0)
        fc_h = []
        bn_h = []
        for i in range(n_depth - 2):
            fc_h.append(L.Linear(n_unit, n_unit, nobias=True, initialW=w))
            bn_h.append(L.BatchNormalization(n_unit))

        super(Generator, self).__init__()
        with self.init_scope():
            self.fc_in = L.Linear(None, n_unit, nobias=True, initialW=w)
            self.bn_in = L.BatchNormalization(n_unit)
            self.fc_h = fc_h
            self.bn_h = bn_h
            self.fc_out = L.Linear(n_unit, cdim, nobias=True, initialW=w)
            self.bn_out = L.BatchNormalization(cdim)

    def __call__(self, x):
        if self.mode == 'Tanh':
            return self.Tanh(x)
        elif self.mode == 'Tanh_BN':
            return self.Tanh_BN(x)
        elif self.mode == 'Softplus':
            return self.Softplus(x)
        elif self.mode == 'Relu':
            return self.Relu(x)

    def Tanh(self, x):
        x = F.tanh(self.fc_in(x))
        for fc_h in self.fc_h:
            x = F.tanh(fc_h(x))
        y = F.sigmoid(self.fc_out(x))

        return y

    def Tanh_BN(self, x):
        x = F.tanh(self.bn_in(self.fc_in(x)))
        for bn_h, fc_h in zip(self.bn_h, self.fc_h):
            x = F.tanh(bn_h(fc_h(x)))
        y = F.sigmoid(self.fc_out(x))

        return y

    def Softplus(self, x):
        x = F.tanh(self.fc_in(x))
        idx = 0
        for fc_h in self.fc_h:
            if idx % 2 == 0:
                x = F.softplus(fc_h(x))
            else:
                x = F.tanh(fc_h(x))
            idx += 1
        y = F.sigmoid(self.fc_out(x))
        return y

    def Relu(self, x):
        x = F.relu(self.fc_in(x))
        for fc_h in self.fc_h:
            x = F.tanh(fc_h(x))
        y = F.sigmoid(self.fc_out(x))

        return y