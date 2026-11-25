from collections import OrderedDict

from torch import nn


class TeacherCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 128), hidden=128, dropout=0.5, n_classes=10):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.widths, self.hidden = tuple(widths), hidden
        self.net = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, w1, 3, padding=1)), ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(w1, w2, 3, padding=1)), ("relu2", nn.ReLU()), ("pool1", nn.MaxPool2d(2)),
            ("conv3", nn.Conv2d(w2, w3, 3, padding=1)), ("relu3", nn.ReLU()),
            ("conv4", nn.Conv2d(w3, w4, 3, padding=1)), ("relu4", nn.ReLU()), ("pool2", nn.MaxPool2d(2)),
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(w4 * 7 * 7, hidden)), ("relu5", nn.ReLU()), ("drop", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden, n_classes)),
        ]))

    def forward(self, x):
        return self.net(x)


class StudentCNN(nn.Module):
    def __init__(self, widths=(16, 32, 32), hidden=56, n_classes=10):
        super().__init__()
        w1, w2, w3 = widths
        self.net = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, w1, 3, padding=1)), ("relu1", nn.ReLU()), ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(w1, w2, 3, padding=1)), ("relu2", nn.ReLU()),
            ("conv3", nn.Conv2d(w2, w3, 3, padding=1)), ("relu3", nn.ReLU()), ("pool2", nn.MaxPool2d(2)),
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(w3 * 7 * 7, hidden)), ("relu4", nn.ReLU()),
            ("fc2", nn.Linear(hidden, n_classes)),
        ]))

    def forward(self, x):
        return self.net(x)


def make_teacher(cfg):
    b = cfg["baseline"]
    return TeacherCNN(b["widths"], b["hidden"], b["dropout"])


def make_student(cfg):
    d = cfg["distill"]
    return StudentCNN(d["widths"], d["hidden"])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def weight_layers(model):
    return [(n, m) for n, m in model.net.named_children() if isinstance(m, (nn.Conv2d, nn.Linear))]


def layer_macs(model, size=28):
    out, hw = {}, size
    for name, m in model.net.named_children():
        if isinstance(m, nn.Conv2d):
            out[name] = m.weight.numel() * hw * hw
        elif isinstance(m, nn.Linear):
            out[name] = m.weight.numel()
        elif isinstance(m, nn.MaxPool2d):
            hw //= 2
    return out


def total_macs(model):
    return sum(layer_macs(model).values())


def effective_macs(model):
    macs = layer_macs(model)
    return sum(macs[n] * (m.weight != 0).sum().item() / m.weight.numel() for n, m in weight_layers(model))


def nonzero_params(model):
    return sum(int((p != 0).sum()) for p in model.parameters())
