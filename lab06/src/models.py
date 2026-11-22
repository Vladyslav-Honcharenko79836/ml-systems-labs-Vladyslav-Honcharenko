from torch import nn


class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, n_classes),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {"CNN": CNN}


def build_model(cfg):
    return MODELS[cfg["model"]["name"]]()
