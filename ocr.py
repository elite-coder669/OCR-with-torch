
from timeit import default_timer as timer
import torch
from torch import nn
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class OCR(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, p: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   
            nn.Dropout(p),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   
            nn.Dropout(p),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   
            nn.Dropout(p),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   
            nn.Dropout(p)
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),               
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} sub-directories and total {len(filenames)} files in the directory: {dirpath}.")

def print_train_time(start: float, end: float, device: torch.device):
    total_time = end - start
    print(f"train time of device : {device}: {total_time:.3f} seconds")
    return total_time

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        running_acc += batch_acc

    avg_loss = running_loss / len(data_loader)
    avg_acc = running_acc / len(data_loader)
    train_loss.append(avg_loss)
    train_accuracy.append(avg_acc)
    print(f"Train loss: {avg_loss:.5f} | Train acc: {avg_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_targets = []
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            running_loss += loss.item()
            batch_acc = accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            running_acc += batch_acc
            all_preds.append(test_pred.argmax(dim=1).cpu())
            all_targets.append(y.cpu())

    avg_loss = running_loss / len(data_loader)
    avg_acc = running_acc / len(data_loader)
    test_loss.append(avg_loss)
    test_accuracy.append(avg_acc)
    print(f"Test loss: {avg_loss:.5f} | Test acc: {avg_acc:.2f}%\n")
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return all_targets, all_preds


train_dir = Path('data/handwritten-english-characters-and-digits/combined_folder/train')
test_dir = Path('data/handwritten-english-characters-and-digits/combined_folder/test')

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    # transforms.RandomRotation(degrees=10),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=1)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=1)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


if __name__ == '__main__':
    from tqdm.auto import tqdm
    walk_through_dir("./data/augmented_images/augmented_images1")
    import random
    from PIL import Image


    image_path = Path('./data/augmented_images/augmented_images1')
    image_path_list = list(image_path.glob('*/*.png'))
    if image_path_list:
        random_image_path = random.choice(image_path_list)
        image_class = random_image_path.parent.stem
        img = Image.open(random_image_path)
        print(f"Random image path: {random_image_path}")
        print(f"Image class: {image_class} | Image size: {img.width}x{img.height}")


    classes = train_dataset.classes
    num_classes = len(classes)
    model_0 = OCR(input_channels=1, num_classes=num_classes, p=0.25).to(device)

    # Print model summary (parameter counts)
    total_params, trainable_params = count_params(model_0)
    print(f"Model total params: {total_params:,} | Trainable params: {trainable_params:,}")
    print(model_0)  # structural summary

    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)  # you can try label_smoothing=0.1 later
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Early stopping variables
    epochs = 150
    patience = 6
    best_acc = 0.0
    trigger_times = 0

    # For reproducibility (best effort)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    start = timer()
    best_preds = None
    best_targets = None

    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f"--------- Epoch {epoch+1}/{epochs} ---------")
        train_step(model=model_0,
                   data_loader=train_dataloader,
                   loss_fn=loss_fn,
                   accuracy_fn=accuracy_fn,
                   optimizer=optimizer,
                   device=device)

        targets, preds = test_step(model=model_0,
                                  data_loader=test_dataloader,
                                  loss_fn=loss_fn,
                                  accuracy_fn=accuracy_fn,
                                  device=device)

        current_acc = test_accuracy[-1]
        scheduler.step(current_acc)  # scheduler based on validation accuracy

        # save best
        if current_acc > best_acc:
            best_acc = current_acc
            trigger_times = 0
            best_preds = preds.copy()
            best_targets = targets.copy()
            torch.save(model_0.state_dict(), 'bestmodel_3.pth')
            print(f"New best test acc: {best_acc:.2f}% (model saved)")
        else:
            trigger_times += 1
            print(f"No improvement. trigger times: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    end = timer()
    print_train_time(start=start, end=end, device=device)


    out_dir = Path("./training_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="train_loss")
    plt.plot(range(1, len(test_loss)+1), test_loss, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Test Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = out_dir / "loss_plot.png"
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to: {loss_plot_path}")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracy)+1), train_accuracy, label="train_acc")
    plt.plot(range(1, len(test_accuracy)+1), test_accuracy, label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train & Test Accuracy")
    plt.legend()
    plt.grid(True)
    acc_plot_path = out_dir / "accuracy_plot.png"
    plt.savefig(acc_plot_path)
    print(f"Saved accuracy plot to: {acc_plot_path}")
    plt.show()


    if best_preds is not None and best_targets is not None:
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(best_targets, best_preds, labels=list(range(num_classes)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax, xticks_rotation='vertical', colorbar=True)
            plt.title("Confusion Matrix")
            cm_path = out_dir / "confusion_matrix.png"
            plt.savefig(cm_path, bbox_inches="tight")
            print(f"Saved confusion matrix to: {cm_path}")
            plt.show()
        except Exception as e:

            print("sklearn not available or plotting failed; falling back to manual confusion matrix. error:", e)
            import numpy as np
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(best_targets, best_preds):
                cm[t, p] += 1
            plt.figure(figsize=(10, 10))
            plt.matshow(cm, fignum=1)
            plt.title("Confusion Matrix (best test)")
            plt.colorbar()
            cm_path = out_dir / "confusion_matrix_fallback.png"
            plt.savefig(cm_path, bbox_inches="tight")
            print(f"Saved fallback confusion matrix to: {cm_path}")
            plt.show()


    del model_0, optimizer
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print("Done.")
