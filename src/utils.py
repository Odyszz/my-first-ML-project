import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Загрузка и обработка данных
def load_and_preprocess_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    x = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    # Разделение данных
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    # Преобразование в numpy
    x_train_np = x_train.to_numpy()
    x_test_np = x_test.to_numpy()
    x_valid_np = x_valid.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    y_valid_np = y_valid.to_numpy()

    # Стандартизация
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_np)
    x_test_scaled = scaler.transform(x_test_np)
    x_valid_scaled = scaler.transform(x_valid_np)

    # Преобразование в тензоры
    x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_valid = torch.tensor(x_valid_scaled, dtype=torch.float32)
    x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
    y_valid = torch.tensor(y_valid_np, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


# Создание dataLoader
def create_dataloaders(x_train, x_valid, x_test, y_train, y_valid, y_test, batch_size):
    print(x_train.shape,x_train[0])
    print(y_train.shape,y_train[0])
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader