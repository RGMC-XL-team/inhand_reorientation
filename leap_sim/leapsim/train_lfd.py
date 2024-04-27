import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from leapsim.learning.lfd_model import LfDAgent
from leapsim.utils.lfd_preprocess import load_and_split_dataset


# 定义模型评估函数
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    return test_loss / len(test_loader.dataset)

# 定义训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    best_test_loss = float('inf')
    train_loss_list, test_loss_list = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        test_loss = evaluate_model(model, test_loader, criterion)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}")

        train_loss_list.append(epoch_loss)
        test_loss_list.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict()
            print(f"saving best model at epoch {epoch}...")
            torch.save(best_model, './debug/lfd/best_rnn_model.pth')

    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='test loss')
    plt.legend()
    plt.show()

# hyper-parameters
include_obj_pose = False
extra_args = {
    "frozen_fingers": ["finger1", "finger3"],
    "free_fingers": ["thumb", "finger2"]
}
input_size = 4 * len(extra_args["free_fingers"]) + 7 if include_obj_pose else 4 * len(extra_args["free_fingers"])
hidden_size = 256
rnn_layers = 1
mlp_hidden_sizes = [512, 256, 128]
output_size = 4 * len(extra_args["free_fingers"])

num_epochs = 100
batch_size = 32
sequence_length = 3

device = "cuda:0"

# load agent
model = LfDAgent(input_size, hidden_size, rnn_layers, mlp_hidden_sizes, output_size)
model.to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# dataset
obses, actions = load_and_split_dataset(sequence_length, extra_args)
obses = torch.tensor(obses, dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.float32).to(device)
train_data, test_data, train_labels, test_labels = train_test_split(obses, actions, test_size=0.2, random_state=42)

if not include_obj_pose:
    train_data = train_data[..., :-7]
    test_data = test_data[..., :-7]

# data loader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)