import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=16 * 54 * 54, out_features=120)  # 调整 in_features
        self.bn3 = nn.BatchNorm1d(120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.bn4 = nn.BatchNorm1d(84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.relu(self.bn1(self.c1(x)))
        x = self.s2(x)
        x = self.relu(self.bn2(self.c3(x)))
        x = self.s4(x)
        # 打印语句仅用于调试
        # print(f"Shape before flatten: {x.shape}")
        x = self.flatten(x)
        x = self.relu(self.bn3(self.f5(x)))
        x = self.relu(self.bn4(self.f6(x)))
        x = self.f7(x)
        return x


# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)
    # 测试输入尺寸为 (1, 224, 224)
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    output = model(dummy_input)
    print(summary(model, input_size=(1, 224, 224)))
