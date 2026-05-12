




import torch
import torch.nn as nn


# =========================================================
# CNN MODEL CLASS
# =========================================================

class CNNModel(nn.Module):

    def __init__(self, dropout_rate=0.3):

        super(CNNModel, self).__init__()


        self.features = nn.Sequential(

            # First Conv Layer
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # Second Conv Layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # =================================================
        # CLASSIFIER
        # =================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 32 * 32, 128),

            nn.ReLU(),

            # Configurable dropout
            nn.Dropout(dropout_rate),

            nn.Linear(128, 2)
        )

    # -----------------------------------------------------
    # FORWARD PASS
    # -----------------------------------------------------
    def forward(self, x):

        x = self.features(x)
        
        x = self.classifier(x)

        return x


# =========================================================
# TEST MODEL
# =========================================================

if __name__ == "__main__":

    model = CNNModel(dropout_rate=0.5)

    print(model)

    dummy_input = torch.randn(32, 3, 128, 128)

    output = model(dummy_input)

    print(f"\nOutput Shape: {output.shape}")