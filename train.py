# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import ResNetClassifier
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0004

# Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)

    # 2. Inisialisasi Model
    model = ResNetClassifier(in_channels=in_channels, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)

    # 3. Mendefinisikan Loss Function dan Optimizer
    # Pilih loss sesuai jumlah kelas
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # CrossEntropyLoss expects logits shape (N, C) and targets LongTensor (N,)
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []

    print("\n--- Memulai Training ---")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            # pindahkan batch ke device
            images = images.to(device)
            labels = labels.to(device)

            # pastikan shape label menjadi (N,1) untuk binary atau (N,) untuk multi-class
            labels = labels.squeeze()  # hilangkan dimensi singleton seperti (N,1,1) -> (N,1) atau (N,)
            if num_classes == 2:
                labels_train = labels.float().unsqueeze(1)   # shape (N,1) untuk BCEWithLogitsLoss
            else:
                labels_train = labels.long()                 # shape (N,) untuk CrossEntropyLoss

            outputs = model(images)
            loss = criterion(outputs, labels_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Hitung training accuracy
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                train_total += labels_train.size(0)
                train_correct += (predicted == labels_train).sum().item()
            else:
                preds = outputs.argmax(dim=1)
                train_total += labels_train.size(0)
                train_correct += (preds == labels_train).sum().item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        train_accuracy = 100 * train_correct / max(1, train_total)

        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # pastikan shape label validasi sama dengan yang dipakai saat training
                labels = labels.squeeze()
                if num_classes == 2:
                    labels_v = labels.float().unsqueeze(1)
                else:
                    labels_v = labels.long()

                outputs = model(images)
                val_loss = criterion(outputs, labels_v)
                val_running_loss += val_loss.item()

                if num_classes == 2:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    val_total += labels_v.size(0)
                    val_correct += (predicted == labels_v).sum().item()
                else:
                    preds = outputs.argmax(dim=1)
                    val_total += labels_v.size(0)
                    val_correct += (preds == labels_v).sum().item()

        avg_val_loss = val_running_loss / max(1, len(val_loader))
        val_accuracy = 100 * val_correct / max(1, val_total)

        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")

    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history,
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
