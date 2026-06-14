# =========================================================
# IMPORT LIBRARIES
# =========================================================

import torch
import torch.nn as nn



class ANNModel(nn.Module):

    def __init__(self):

        # Initialize parent class
        super(ANNModel, self).__init__()

        self.model = nn.Sequential(

            # FLATTEN LAYER
            nn.Flatten(),

            # FIRST FULLY CONNECTED LAYER
            nn.Linear(3 * 128 * 128, 512),

            # Activation function
            nn.ReLU(),

            # Dropout helps prevent overfitting
            nn.Dropout(0.3),

            # SECOND FULLY CONNECTED LAYER
            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            # OUTPUT LAYER
            nn.Linear(128, 2)
        )

    # FORWARD PASS\
    def forward(self, x):

        return self.model(x)



if __name__ == "__main__":

    model = ANNModel()

    # Print architecture
    print(model)

    # CREATE DUMMY INPUT
    dummy_input = torch.randn(32, 3, 128, 128)

    # FORWARD PASS
    output = model(dummy_input)

    # OUTPUT SHAPE
    print(f"\nOutput Shape: {output.shape}")

    # Expected:
    # torch.Size([32, 2])