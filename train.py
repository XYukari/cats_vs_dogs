import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import dataloaders
from model import build_model
from settings import device, num_epochs, patience, learning_rate, model_path

def train_model():
    model = build_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(dataloaders["train"], desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloaders["train"])
        print(f"  Train Loss: {avg_train_loss:.4f}")

        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloaders["val"], desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataloaders["val"])
        print(f"  Val Loss: {avg_val_loss:.4f}")

        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
            print("  Saved new best model.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break

    print("Training Complete!")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
