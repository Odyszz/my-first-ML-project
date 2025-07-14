import torch
import torch.nn as nn

class HousingNet(nn.Module):
    def __init__(self):
        super(HousingNet, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # Вход: 8 признаков, выход: 32 нейрона
        self.fc2 = nn.Linear(32, 32) # Вход: 32, выход: 32
        self.fc3 = nn.Linear(32, 16) # Вход: 32, выход: 16
        self.fc4 = nn.Linear(16, 1)  # Выход: 1 (цена дома)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Активация ReLU
        x = torch.relu(self.fc2(x))  # Активация ReLU
        x = torch.relu(self.fc3(x))  # Активация ReLU
        x = self.fc4(x)  # Линейный выход (без активации для регрессии)
        return x
