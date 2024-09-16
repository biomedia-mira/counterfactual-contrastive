from torchvision import models
import torch
from torch.nn.functional import normalize


class ResNetBase(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_name: str,
        pretrained: bool,
        input_channels: int = 3,
        normalise_features: bool = False,
    ) -> None:
        super().__init__()
        match encoder_name:
            case "resnet50":
                if pretrained:
                    self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                else:
                    self.net = models.resnet50(weights=None)
            case "resnet18":
                if pretrained:
                    self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                else:
                    self.net = models.resnet18(weights=None)
            case "resnet152":
                if pretrained:
                    self.net = models.resnet152(
                        weights=models.ResNet152_Weights.DEFAULT
                    )
                else:
                    self.net = models.resnet152(weights=None)
            case _:
                raise ValueError(f"Encoder name {encoder_name} not recognised.")
        self.num_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(self.num_features, num_classes)
        self.num_classes = None
        self.normalise_features = normalise_features
        if input_channels != 3:
            self.net.conv1 = torch.nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x0 = self.net.maxpool(x)
        x1 = self.net.layer1(x0)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x4 = self.net.avgpool(x4)
        x4 = torch.flatten(x4, 1)
        if self.normalise_features:
            x4 = normalize(x4, dim=1)
        return x4

    def classify_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)

    def reset_classifier(self, num_classes):
        self.net.fc = torch.nn.Linear(self.num_features, num_classes)
        for p in self.net.fc.parameters():
            p.requires_grad = True
