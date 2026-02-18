import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import traceback

# ========================== CONFIGURATION ==========================
# Set your folder paths here
A_DIR = r"C:\Users\mhdi\Downloads\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\train\A"
B_DIR = r"C:\Users\mhdi\Downloads\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\train\B"
MASK_DIR = r"C:\Users\mhdi\Downloads\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\train\OUT"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 4          # adjust based on your GPU memory
EPOCHS = 15              # you can increase for better convergence
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False
NUM_WORKERS = 4          # number of CPU threads for data loading
POS_WEIGHT = 1.0         # weight for the positive class (change). Increase if change is rare.

# ========================== DATASET ==========================
class ChangeDetectionDataset(Dataset):
    """Loads triplets (A, B, mask) from three directories, matching by base name."""
    def __init__(self, a_dir, b_dir, mask_dir, transform=None, mask_transform=None):
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Use files in A as reference
        self.filenames = sorted([f for f in os.listdir(a_dir) if not f.startswith('.')])
        print(f"[Dataset] Found {len(self.filenames)} files in A directory.")
        print(f"[Dataset] Files in B: {len(os.listdir(b_dir))}, Files in mask: {len(os.listdir(mask_dir))}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        base, ext = os.path.splitext(fname)

        # Paths for A and B (assume same filename)
        a_path = os.path.join(self.a_dir, fname)
        b_path = os.path.join(self.b_dir, fname)

        # Find mask file (may have different extension)
        mask_path = os.path.join(self.mask_dir, fname)
        if not os.path.exists(mask_path):
            for ext_candidate in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
                candidate = os.path.join(self.mask_dir, base + ext_candidate)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            else:
                raise FileNotFoundError(f"No mask file found for {fname} in {self.mask_dir}")

        # Load images
        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')   # grayscale mask

        # Apply transforms
        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Concatenate A and B along channel dimension → (6, H, W)
        x = torch.cat([a_img, b_img], dim=0)
        # Mask: (1, H, W) → (H, W) as long (0 or 1)
        y = mask.squeeze(0).long()

        return x, y

# ========================== U‑NET MODEL ==========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ========================== METRICS (FIXED) ==========================
def iou_score(pred, target, smooth=1e-6):
    """IoU = intersection / union, threshold at 0.5."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + smooth) / (union + smooth)

def accuracy(pred, target):
    """Pixel accuracy, threshold at 0.5."""
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / target.numel()

def f1_score(pred, target, smooth=1e-6):
    """F1 score (Dice) for binary masks."""
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    return f1

# ========================== TRAINING FUNCTIONS ==========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    num_batches = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)                          # (B, 1, H, W)
        loss = criterion(logits.squeeze(1), y.float())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.sigmoid(logits)
            total_loss += loss.item()
            total_iou += iou_score(pred, y.unsqueeze(1)).item()
            total_acc += accuracy(pred, y.unsqueeze(1)).item()
            total_f1 += f1_score(pred, y.unsqueeze(1)).item()

        if batch_idx % 50 == 0:   # print every 50 batches
            print(f"  Batch {batch_idx:3d}/{num_batches} | Loss: {loss.item():.4f}")

    n = num_batches
    return total_loss/n, total_iou/n, total_acc/n, total_f1/n

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.squeeze(1), y.float())
            pred = torch.sigmoid(logits)
            total_loss += loss.item()
            total_iou += iou_score(pred, y.unsqueeze(1)).item()
            total_acc += accuracy(pred, y.unsqueeze(1)).item()
            total_f1 += f1_score(pred, y.unsqueeze(1)).item()
    n = len(loader)
    return total_loss/n, total_iou/n, total_acc/n, total_f1/n

# ========================== VISUALIZATION ==========================
def plot_training_curves(train_loss, val_loss, train_iou, val_iou, train_f1, val_f1):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 3, 2)
    plt.plot(train_iou, label='Train')
    plt.plot(val_iou, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU Curves')

    plt.subplot(1, 3, 3)
    plt.plot(train_f1, label='Train')
    plt.plot(val_f1, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Curves')

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_curves.png'))
    plt.show()

def visualize_sample(model, dataset, device, idx=0):
    model.eval()
    x, y = dataset[idx]
    x_batch = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_batch)
        pred = torch.sigmoid(logits).cpu().numpy().squeeze()
    pred_mask = (pred > 0.5).astype(np.uint8)

    a_img = x[:3].cpu().numpy().transpose(1,2,0)
    b_img = x[3:].cpu().numpy().transpose(1,2,0)
    gt_mask = y.cpu().numpy().astype(np.uint8)

    a_img = np.clip(a_img, 0, 1)
    b_img = np.clip(b_img, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(a_img)
    plt.title('Image A')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(b_img)
    plt.title('Image B')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, f'sample_{idx}_comparison.png'))
    plt.show()

# ========================== MAIN ==========================
def main():
    try:
        print("=" * 60)
        print("CHANGE DETECTION TRAINING SCRIPT")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {DEVICE}")
        print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
        print(f"Checkpoint dir: {CHECKPOINT_DIR}")

        # Check directories
        for d in [A_DIR, B_DIR, MASK_DIR]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Directory not found: {d}")
        print("All directories exist.")

        # Data transforms
        transform = transforms.Compose([transforms.ToTensor()])
        mask_transform = transforms.Compose([transforms.ToTensor()])

        # Create dataset
        print("\nCreating dataset...")
        full_dataset = ChangeDetectionDataset(A_DIR, B_DIR, MASK_DIR,
                                              transform=transform,
                                              mask_transform=mask_transform)
        print(f"Total samples: {len(full_dataset)}")

        # Split train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print(f"Train samples: {train_size}, Val samples: {val_size}")

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Model
        print("\nBuilding U-Net...")
        model = UNet(n_channels=6, n_classes=1).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Loss and optimizer
        pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        print("\nStarting training...")
        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        train_ious, val_ious = [], []
        train_f1s, val_f1s = [], []

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            print("-" * 40)

            train_loss, train_iou, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, epoch)
            val_loss, val_iou, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            # Print summary
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train IoU : {train_iou:.4f} | Val IoU : {val_iou:.4f}")
            print(f"Train F1  : {train_f1:.4f} | Val F1  : {val_f1:.4f}")
            print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
                print(f"  >>> Best model saved (val loss = {val_loss:.4f})")

        print("\nTraining completed!")

        # Plot curves
        print("Plotting training curves...")
        plot_training_curves(train_losses, val_losses, train_ious, val_ious, train_f1s, val_f1s)

        # Test on a validation sample
        print("Testing on a validation sample...")
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth')))
        visualize_sample(model, val_dataset, DEVICE, idx=0)

        print("\nAll done! Check the 'checkpoints' folder for outputs.")

    except Exception as e:
        print("\n" + "!" * 60)
        print("AN ERROR OCCURRED:")
        traceback.print_exc()
        print("!" * 60)

if __name__ == "__main__":
    main()