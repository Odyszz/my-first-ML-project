import torch
import torch.nn as nn
import torch.optim as optim


# Оценка модели
def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    model.train()
    return avg_loss


# Обучение модели
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    best_val = float('inf')
    for epoch in range(num_epochs+1):
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        # Вычисляем val_loss на каждой эпохе
        val_loss = evaluate(valid_loader, model, criterion)

        # Печатаем каждые 10 эпох
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')


# Тестирование модели
def test_model(model, test_loader, criterion):
    model.eval()
    total_mse = 0
    num_batches = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            mse = criterion(outputs, labels)
            total_mse += mse.item()
            num_batches += 1
    avg_mse = total_mse / num_batches
    print(f"Average Test MSE: {avg_mse:.2f}")