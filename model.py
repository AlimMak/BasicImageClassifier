import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Part 1: find patterns in the image
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3 color channels in, 32 pattern maps out
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32 in, 64 out
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16x16 -> 8x8
        )
        # Part 2: turn those patterns into 10 scores
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),               # randomly turn off half these neurons during training
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = SimpleCNN()
    fake_batch = torch.randn(64, 3, 32, 32)   # 64 random "images"
    out = model(fake_batch)
    print("output shape:", out.shape)
    print("parameters:", sum(p.numel() for p in model.parameters()))