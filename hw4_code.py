import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.init import kaiming_normal_, constant_

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# =============================================
# Q1: Baseline CNN Model
# =============================================
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =============================================
# Q2: Architectural Variants
# =============================================
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*7*7, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class KernelSizeCNN(nn.Module):
    def __init__(self):
        super(KernelSizeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.lrelu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SkipCNN(nn.Module):
    def __init__(self):
        super(SkipCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.tanh3 = nn.Tanh()
        
        # Skip connection from conv1 to conv3
        # We need to adjust the channels (32->64) and spatial dimensions (28x28->7x7)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=4, stride=4)  # Reduce spatial dimensions from 28x28 to 7x7
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.tanh1(x1)
        x = self.pool1(x1)  # 28x28 -> 14x14
        
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)  # 14x14 -> 7x7
        
        x = self.conv3(x)
        x_skip = self.skip_conv(x1)  # Process skip connection to match dimensions
        x = x + x_skip  # Now both are 64 channels and 7x7 spatial
        x = self.tanh3(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# =============================================
# Q3: Hyperparameter Analysis Models
# =============================================
class HyperparamCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, init_method='default'):
        super(HyperparamCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)
        
        # Weight initialization
        if init_method == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =============================================
# Q4: Failed Experiment Models
# =============================================
class NoActivationCNN(nn.Module):
    def __init__(self):
        super(NoActivationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class NoPoolingCNN(nn.Module):
    def __init__(self):
        super(NoPoolingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class TooSmallCNN(nn.Module):
    def __init__(self):
        super(TooSmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2*14*14, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# =============================================
# Training and Evaluation Functions
# =============================================
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, accuracy

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

def train_and_evaluate(model, name, train_loader, test_loader, epochs=10, 
                      lr=0.001, optimizer_type='Adam', batch_size=64, 
                      plot_results=True):
    print(f"\n=== Training {name} ===")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    if plot_results:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{name} Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'{name} Accuracy Curves')
        plt.show()
    
    return {
        'name': name,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'model': model
    }

# =============================================
# Main Execution
# =============================================
def main():
    # Create data loaders with default batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # =========================================
    # Q1: Baseline CNN
    # =========================================
    print("\n" + "="*50)
    print("QUESTION 1: BASELINE CNN")
    print("="*50)
    baseline_results = train_and_evaluate(
        BaselineCNN(), "Baseline CNN", train_loader, test_loader)
    
    # =========================================
    # Q2: Architectural Variants
    # =========================================
    print("\n" + "="*50)
    print("QUESTION 2: ARCHITECTURAL VARIANTS")
    print("="*50)
    
    # Train all variants
    deeper_results = train_and_evaluate(
        DeeperCNN(), "Deeper CNN with BatchNorm", train_loader, test_loader)
    kernel_results = train_and_evaluate(
        KernelSizeCNN(), "Kernel Size CNN", train_loader, test_loader)
    skip_results = train_and_evaluate(
        SkipCNN(), "Skip Connection CNN", train_loader, test_loader)
    
    # Compare all models from Q1 and Q2
    all_results = [baseline_results, deeper_results, kernel_results, skip_results]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for res in all_results:
        plt.plot(res['train_accs'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for res in all_results:
        plt.plot(res['test_accs'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for res in all_results:
        plt.plot(res['train_losses'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for res in all_results:
        plt.plot(res['test_losses'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # =========================================
    # Q3: Hyperparameter Analysis
    # =========================================
    print("\n" + "="*50)
    print("QUESTION 3: HYPERPARAMETER ANALYSIS")
    print("="*50)
    
    # Learning Rate Analysis
    print("\nLearning Rate Analysis:")
    lrs = [0.1, 0.01, 0.001, 0.0001]
    lr_results = []
    for lr in lrs:
        model = HyperparamCNN(dropout_rate=0.5)
        res = train_and_evaluate(
            model, f"LR={lr}", train_loader, test_loader, epochs=5, lr=lr, plot_results=False)
        lr_results.append(res)
        print(f"Final Test Acc (LR={lr}): {res['test_accs'][-1]:.2f}%")
    
    # Optimizer Analysis
    print("\nOptimizer Analysis:")
    optimizers = ['Adam', 'SGD']
    opt_results = []
    for opt in optimizers:
        model = HyperparamCNN(dropout_rate=0.5)
        res = train_and_evaluate(
            model, f"Optimizer={opt}", train_loader, test_loader, epochs=5, 
            optimizer_type=opt, plot_results=False)
        opt_results.append(res)
        print(f"Final Test Acc ({opt}): {res['test_accs'][-1]:.2f}%")
    
    # Dropout Rate Analysis
    print("\nDropout Rate Analysis:")
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    dropout_results = []
    for rate in dropout_rates:
        model = HyperparamCNN(dropout_rate=rate)
        res = train_and_evaluate(
            model, f"Dropout={rate}", train_loader, test_loader, epochs=5, plot_results=False)
        dropout_results.append(res)
        print(f"Final Test Acc (Dropout={rate}): {res['test_accs'][-1]:.2f}%")
    
    # Weight Initialization Analysis
    print("\nWeight Initialization Analysis:")
    init_methods = ['default', 'kaiming']
    init_results = []
    for method in init_methods:
        model = HyperparamCNN(init_method=method)
        res = train_and_evaluate(
            model, f"Init={method}", train_loader, test_loader, epochs=5, plot_results=False)
        init_results.append(res)
        print(f"Final Test Acc (Init={method}): {res['test_accs'][-1]:.2f}%")
    
    # =========================================
    # Q4: Failed Experiments
    # =========================================
    print("\n" + "="*50)
    print("QUESTION 4: FAILED EXPERIMENTS")
    print("="*50)
    
    # Experiment 1: No Activation Functions
    print("\nExperiment 1: No Activation Functions")
    no_act_results = train_and_evaluate(
        NoActivationCNN(), "No Activation CNN", train_loader, test_loader, epochs=5)
    
    # Experiment 2: No Pooling Layers
    print("\nExperiment 2: No Pooling Layers")
    no_pool_results = train_and_evaluate(
        NoPoolingCNN(), "No Pooling CNN", train_loader, test_loader, epochs=5)
    
    # Experiment 3: Too Small Network
    print("\nExperiment 3: Too Small Network")
    small_results = train_and_evaluate(
        TooSmallCNN(), "Too Small CNN", train_loader, test_loader, epochs=5)
    
    # Compare failed experiments
    failed_results = [no_act_results, no_pool_results, small_results]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for res in failed_results:
        plt.plot(res['train_accs'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Failed Experiments - Training Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for res in failed_results:
        plt.plot(res['test_accs'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Failed Experiments - Test Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for res in failed_results:
        plt.plot(res['train_losses'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Failed Experiments - Training Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for res in failed_results:
        plt.plot(res['test_losses'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Failed Experiments - Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
