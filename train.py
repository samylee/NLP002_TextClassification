import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloder


class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    text_path = 'data/train.txt'
    train_loader, test_loader, labels = get_dataloder(text_path)

    input_dim = 100
    hidden_dim = 50
    output_dim = len(set(labels))
    model = TextClassifier(input_dim, hidden_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs} - Batch: {batch_idx}/{len(train_loader)} - Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy of the model on the test data: {100 * correct / total}%")


if __name__ == '__main__':
    train()