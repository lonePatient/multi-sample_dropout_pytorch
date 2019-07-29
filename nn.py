import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes,dropout_num,dropout_p):
        super(ResNet, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.fc = nn.Linear(128,num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x,y = None,loss_fn = None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feature = F.avg_pool2d(out, 4)
        if len(self.dropouts) == 0:
            out = feature.view(feature.size(0), -1)
            out = self.fc(out)
            if loss_fn is not None:
                loss = loss_fn(out,y)
                return out,loss
            return out,None
        else:
            for i,dropout in enumerate(self.dropouts):
                if i== 0:
                    out = dropout(feature)
                    out = out.view(out.size(0),-1)
                    out = self.fc(out)
                    if loss_fn is not None:
                        loss = loss_fn(out, y)
                else:
                    temp_out = dropout(feature)
                    temp_out = temp_out.view(temp_out.size(0),-1)
                    out =out+ self.fc(temp_out)
                    if loss_fn is not None:
                        loss = loss+loss_fn(temp_out, y)
            if loss_fn is not None:
                return out / len(self.dropouts),loss / len(self.dropouts)
            return out,None

def MiniResNet(num_classes=10,dropout_num = 0,dropout_p = 0.5):
    return ResNet(ResidualBlock,num_classes=num_classes,dropout_num = dropout_num,dropout_p = dropout_p)

