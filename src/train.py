# =========================================================
# train.py
# Hazelnut Classification - Reusable Training Pipeline
# =========================================================

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader

from model_ANN import ANNModel
from model_CNN import CNNModel


# =========================================================
# TRAIN MODEL FUNCTION
# =========================================================

def train_model(
    model_type="cnn",
    epochs=10,
    learning_rate=0.001,
    optimizer_name="adam",
    dropout_rate=0.3,
    weight_decay=0.0001
):

    # =====================================================
    # DEVICE SETUP
    # =====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(f"Using Device: {device}")
    print("=" * 60)


    # =====================================================
    # CREATE FOLDERS
    # =====================================================

    os.makedirs("notebooks/models", exist_ok=True)

    os.makedirs("notebooks/results", exist_ok=True)

    results_file="notebooks/results/experiment_results.csv"


 


    # =====================================================
    # MODEL CREATION
    # =====================================================

    if model_type.lower() == "ann":

        MODEL_NAME = "ann"

        model = ANNModel()

    elif model_type.lower() == "cnn":

        MODEL_NAME = "cnn"

        model = CNNModel(
            dropout_rate=dropout_rate
        )

    else:
        raise ValueError("Invalid model type. Use 'ann' or 'cnn'.")


    # Move model to device
    model = model.to(device)


    # =====================================================
    # LOSS FUNCTION
    # =====================================================

    criterion = nn.CrossEntropyLoss()


    # =====================================================
    # OPTIMIZER SELECTION
    # =====================================================

    if optimizer_name.lower() == "adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    elif optimizer_name.lower() == "sgd":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif optimizer_name.lower() == "rmsprop":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    else:
        raise ValueError("Invalid optimizer selected.")


    # =====================================================
    # TRAINING START
    # =====================================================

    print("\nStarting Training...\n")


    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(epochs):

        # =================================================
        # TRAIN MODE
        # =================================================

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        # =================================================
        # TRAINING BATCHES
        # =================================================

        for images, labels in train_loader:

            # Move data to device
            images = images.to(device)

            labels = labels.to(device)


            # =============================================
            # ANN INPUT RESHAPE
            # =============================================

            if MODEL_NAME == "ann":

                images = images.view(images.size(0), -1)


            # =============================================
            # FORWARD PASS
            # =============================================

            outputs = model(images)

            loss = criterion(outputs, labels)


            # =============================================
            # BACKPROPAGATION
            # =============================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # =============================================
            # TRAIN METRICS
            # =============================================

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


        # =================================================
        # FINAL TRAIN RESULTS
        # =================================================

        train_loss = running_loss / len(train_loader)

        train_acc = 100 * correct / total


        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)


                # =========================================
                # ANN INPUT RESHAPE
                # =========================================

                if MODEL_NAME == "ann":

                    images = images.view(images.size(0), -1)


                # =========================================
                # FORWARD PASS
                # =========================================

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (predicted == labels).sum().item()


        # =================================================
        # FINAL VALIDATION RESULTS
        # =================================================

        val_loss = val_loss / len(val_loader)

        val_acc = 100 * val_correct / val_total


        # =================================================
        # PRINT RESULTS
        # =================================================

        print("=" * 60)

        print(f"Epoch [{epoch + 1}/{epochs}]")

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Val Acc    : {val_acc:.2f}%")


        # =================================================
        # OVERFITTING CHECK
        # =================================================

        if train_acc - val_acc > 15:

            print("Possible Overfitting Detected")

        print("=" * 60)


    # =====================================================
    # MODEL SAVE NAME
    # =====================================================

    model_name = (
        f"{MODEL_NAME}_"
        f"opt-{optimizer_name}_"
        f"lr-{learning_rate}_"
        f"drop-{dropout_rate}_"
        f"wd-{weight_decay}_"
        f"epoch-{epochs}.pth"
    )

    save_path = os.path.join("notebooks/models", model_name)


    # =====================================================
    # SAVE MODEL
    # =====================================================

    torch.save(model.state_dict(), save_path)

    print("\nModel Saved Successfully")

    print(f"Saved Path: {save_path}")


    # =====================================================
    # SAVE RESULTS TO CSV
    # =====================================================

    file_exists = os.path.isfile(results_file)

    try:

        with open(results_file, mode="a", newline="") as file:

            writer = csv.writer(file)

            # =============================================
            # WRITE HEADER ONLY ONCE
            # =============================================

            if not file_exists:

                writer.writerow([
                    "Model",
                    "Optimizer",
                    "Learning Rate",
                    "Dropout",
                    "Weight Decay",
                    "Epochs",
                    "Train Loss",
                    "Train Accuracy",
                    "Validation Loss",
                    "Validation Accuracy",
                    "Model Path"
                ])


            # =============================================
            # WRITE EXPERIMENT DATA
            # =============================================

            writer.writerow([
                MODEL_NAME,
                optimizer_name,
                learning_rate,
                dropout_rate,
                weight_decay,
                epochs,
                round(train_loss, 4),
                round(train_acc, 2),
                round(val_loss, 4),
                round(val_acc, 2),
                save_path
            ])

    except PermissionError:

        print("\nClose experiment_results.csv before training.")


    # =====================================================
    # TRAINING COMPLETE
    # =====================================================

    print("\n" + "=" * 60)

    print("Training Finished Successfully")

    print("=" * 60)


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {
        "model": MODEL_NAME,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "model_path": save_path
    }


# =========================================================
# DIRECT SCRIPT EXECUTION
# =========================================================

if __name__ == "__main__":

    train_model(
        model_type="cnn",
        epochs=10,
        learning_rate=0.001,
        optimizer_name="adam",
        dropout_rate=0.3,
        weight_decay=0.0001
    )