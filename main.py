#импорты
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import random

from sklearn.datasets import fetch_california_housing #датасет
from sklearn.model_selection import train_test_split # разбиение датасета
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#загрузка
data = fetch_california_housing(as_frame=True)
df = data.frame

#разделение цены и признаков
aftersign = df.drop(columns="MedHouseVal")   # все столбцы, кроме цены(8 признаков)
avgprice = df["MedHouseVal"]       # цена
#print(aftersign.info())


#разделение данных на тестовые и тренировочные
x_train, x_test, y_train, y_test = train_test_split(aftersign, avgprice, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
#to_numpy
x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()
x_valid_np = x_valid.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()
y_valid_np = y_valid.to_numpy()


#standartiziruem
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train_np)
X_test_scaled = scaler.transform(x_test_np)
X_valid_scaled = scaler.transform(x_valid_np)


# Преобразование в тензоры PyTorch
'''
первые проблемы с формой данных в y_train и y_test решились с помощью reshape
'''

x_train = torch.tensor(X_train_scaled, dtype=torch.float32)
x_valid = torch.tensor(X_valid_scaled, dtype=torch.float32)
x_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
y_valid = torch.tensor(y_valid_np, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)


# Создаём датасеты
train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

#разбиваем на батчи
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#функция оценки
def evaluate(loader, model, criterion):
    model.eval()                    # переводим в режим оценки
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    model.train()                   # возвращаем в режим тренировки
    return avg_loss

#модель нейросети
class HousingNet(nn.Module):
    def __init__(self):
        super(HousingNet, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # Первый слой
        self.fc2 = nn.Linear(32, 32) # Второй слой
        self.fc3 = nn.Linear(32, 16)  # Третий слой
        self.fc4 = nn.Linear(16, 1)  # четвертый слой
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # активируем relu
        x = torch.relu(self.fc2(x)) # активируем relu
        x = torch.relu(self.fc3(x)) # активируем relu
        x = self.fc4(x) # не активируем relu тк нужна цена, а не вероятность
        return x

pred_price_net = HousingNet()
#pred_price_net = pred_price_net.to(device)#GPU

optimizer = optim.Adam(pred_price_net.parameters(), lr=0.001)# запускаем оптимизатор
criterion = nn.MSELoss()# запускаем расчет потерь

#обучение
best_val = float('inf')
train_mod = float('inf')
val_mod = float('inf')
for epoch in range(50):
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()#очищение градиентов
        outputs = pred_price_net(inputs) # получение результатов модели
        loss = criterion(outputs, labels)# расчет потерь
        loss.backward()#расчет градиентов для каждого веса
        optimizer.step()# шаг обучения
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss / len(train_dataset)

    if epoch % 10 == 0:
        val_loss = evaluate(valid_loader, pred_price_net,criterion)
        print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}")

    if val_loss < best_val: #Сохраняем лучшую модель
        best_val = val_loss
        torch.save(pred_price_net.state_dict(), 'best_model.pth')

#тестирование
with torch.no_grad():
    total_mse = 0
    num_batches = 0
    for inputs, labels in test_loader:
        outputs = pred_price_net(inputs)
        mse = criterion(outputs, labels)
        total_mse += mse.item()
        num_batches += 1
    avg_mse = total_mse / num_batches
    print(f"Average Test MSE: {avg_mse:.2f}")