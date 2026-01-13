import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Import từ các file bạn đã tạo
from cnn_baseline import get_model as get_baseline
from cnn_transfer import get_model as get_transfer
from vit_timm import get_model as get_vit

def train_one_model(model_name, model, train_loader, val_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    print(f"\n>>> Đang train: {model_name} trên {device}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Acc: {epoch_acc:.2f}%")
        
        # Lưu file best.pt vào thư mục output riêng như yêu cầu
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            output_dir = f"output_{model_name.lower()}"
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))

if __name__ == "__main__":
    # ĐƯỜNG DẪN KHỚP VỚI ẢNH CỦA BẠN (D:/AI/dataset)
    data_dir = 'D:/AI/dataset' 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Tự động load từ thư mục train và val
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    num_classes = len(train_ds.classes)
    print(f"Phát hiện {num_classes} lớp: {train_ds.classes}")

    # CHẠY ĐỦ 3 MODEL THEO THỨ TỰ KHUYẾN NGHỊ TRONG ẢNH
    models_to_run = [
        ("cnn_baseline", get_baseline(num_classes)),
        ("cnn_transfer", get_transfer(num_classes)),
        ("vit", get_vit(num_classes))
    ]

    for name, model in models_to_run:
        train_one_model(name, model, train_loader, val_loader, epochs=5)

    print("\n--- TẤT CẢ ĐÃ DONE! Kiểm tra các thư mục output_... để lấy best.pt ---")