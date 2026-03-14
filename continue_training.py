
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ocr import OCR, train_step, test_step, accuracy_fn
from pathlib import Path
from ocr import train_accuracy, test_accuracy,test_loss,train_loss
if __name__ == '__main__':
    # Device setup (same as before)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dir = Path('./datasets/sujaymann/handwritten-english-characters-and-digits/versions/6/handwritten-english-characters-and-digits/combined_folder/train')
    test_dir = Path('./datasets/sujaymann/handwritten-english-characters-and-digits/versions/6/handwritten-english-characters-and-digits/combined_folder/test')


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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Model instantiation
    num_classes = len(train_dataset.classes)
    model = OCR(input_channels=1, num_classes=num_classes, p=0.33).to(device)

    # Load saved weights
    checkpoint_path = Path("./bestmodel.pth")
    if checkpoint_path.is_file():
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded checkpoint weights from bestmodel.pth")
    else:
        print("Checkpoint not found, training from scratch")

    # Loss, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Continue training parameters
    epochs_to_continue = 20
    patience = 6
    best_acc = 0.0
    trigger_times = 0


    for epoch in range(epochs_to_continue):
        print(f"--------- Continuing Epoch {epoch+1}/{epochs_to_continue} ---------")
        train_step(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device)
        targets, preds = test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

        current_acc = test_accuracy[-1]
        scheduler.step(current_acc)
        if current_acc > best_acc:
            best_acc = current_acc
            trigger_times = 0
            torch.save(model.state_dict(), 'bestmodel.pth')
