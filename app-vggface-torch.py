import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
from PIL import Image

# 数据路径
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
validation_dataset = datasets.ImageFolder(validation_dir, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# 加载 VGGFace 模型，不包括顶部的全连接层
vggface = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
base_model = nn.Sequential(*list(vggface.children())[:-2])

# 定义自定义的顶层
class VGGFaceNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(VGGFaceNet, self).__init__()
        self.base_model = base_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

num_classes = len(train_dataset.classes)
model = VGGFaceNet(base_model, num_classes)

# 冻结卷积基
for param in model.base_model.parameters():
    param.requires_grad = False

# 编译模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Epoch {epoch+1}/{50}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'my_vggface_model.pth')

# 单张图片预测
def predict_image(img_path):
    model.eval()
    img = Image.open(img_path)
    img = data_transforms['val'](img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class_prob = torch.softmax(outputs, 1)[0][predicted].item()

    print(f'Predicted class index: {predicted.item()}')
    print(f'Confidence: {predicted_class_prob:.2f}')

# 示例使用
# predict_image('path_to_your_image.jpg')  # 替换为你要预测的图片路径
