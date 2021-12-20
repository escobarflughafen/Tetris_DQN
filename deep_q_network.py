import torch.nn as nn

class DeepQNetwork2(nn.Module):
    def __init__(self):
        super(DeepQNetwork2, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(1+1+10+9+10, 128), nn.ReLU(inplace=True))        # 5 for state
        self.layer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Linear(128, 1))

        self._create_weights()

    # initialize weights by xavier_uniform
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class DeepQNetwork1(nn.Module):
    def __init__(self):
        super(DeepQNetwork1, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(5, 80), nn.ReLU(inplace=True))        # 5 for state
        self.layer2 = nn.Sequential(nn.Linear(80, 80), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(80, 1))

        self._create_weights()

    # initialize weights by xavier_uniform
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        return x

