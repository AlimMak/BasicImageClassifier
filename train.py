import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import SimpleCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training images get random flips and shifts so the model can't memorize exact pictures
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
plain_transform = transforms.ToTensor()

aug_train = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
plain_train = datasets.CIFAR10(root="data", train=True, download=True, transform=plain_transform)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=plain_transform)

# Same 45,000 / 5,000 split, but validation images are never augmented
indices = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
train_data = Subset(aug_train, indices[:45000])
val_data = Subset(plain_train, indices[45000:])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)

model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)   # highest score = the guess
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

best_val = 0
for epoch in range(1, 26):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        loss = loss_fn(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc = accuracy(train_loader)
    val_acc = accuracy(val_loader)
    note = ""
    if val_acc > best_val:            # early stopping: only keep the best version
        best_val = val_acc
        torch.save(model.state_dict(), "model.pt")
        note = "  <- best, saved"
    print(f"epoch {epoch:2d} | loss {total_loss/len(train_loader):.3f} | "
          f"train {train_acc:.1f}% | val {val_acc:.1f}%{note}")

model.load_state_dict(torch.load("model.pt"))
print(f"best val: {best_val:.1f}% | final test accuracy: {accuracy(test_loader):.1f}%")