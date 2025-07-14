import torch
import torch.optim as optim
import numpy as np
import random
from src.model import HousingNet
from src.utils import load_and_preprocess_data, create_dataloaders
from src.train import train_model, test_model


seed = 42
batch_size = 18
lr = 0.0005
num_epochs = 80

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Загрузка данных
x_train, x_valid, x_test, y_train, y_valid, y_test = load_and_preprocess_data()

# Создание dataLoader
train_loader, valid_loader, test_loader = create_dataloaders(x_train, x_valid, x_test, y_train, y_valid, y_test, batch_size)

# Инициализация модели
model = HousingNet()
criterion = torch.nn.MSELoss()
optimizer = optim.RAdam(model.parameters(), lr=lr)

# Обучение
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pth'))

# Тестирование
test_model(model, test_loader, criterion)
