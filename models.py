from logging.config import valid_ident
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from resnet_model import ResNet50  # 确保这里正确地导入了你的ResNet50模型
from torchvision import models
import copy
import torch.nn.functional as F


# def get_resnet101_model(num_classes, use_se=True):
#     model = models.resnet101(pretrained=True)
#     num_features = model.fc.in_features

#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 4096),  
#         nn.ReLU(),
#         nn.BatchNorm1d(4096),           
#         nn.Dropout(0.4),                
#         nn.Linear(4096, 2048),          
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(2048, num_classes)              
#     )
#     return model

class CustomResNet101(nn.Module):
    def __init__(self, num_classes, use_se=True):
        super(CustomResNet101, self).__init__()
        # 加载预训练的 resnet101 模型
        self.resnet101 = models.resnet101(pretrained=True)
        num_features = self.resnet101.fc.in_features

        # 替换原始的全连接层
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        logits = self.resnet101(x)
        #probs = F.softmax(logits, dim=1)
        return logits

def get_resnet101_model(num_classes, use_se=True):
    model = CustomResNet101(2, use_se)
    return model
# if __name__ == "__main__":
#     model = get_resnet101_model(in_channels=3, num_classes=2, use_se=True)
# class CustomResNet101(nn.Module):
#     def __init__(self, in_channels=3, num_classes=2, use_se=True ):
#         super(CustomResNet101, self).__init__()
#         #self.resnet101 = models.resnet101(pretrained=True)
#         model = models.resnet101(pretrained=True)
#         num_features = model.fc.in_features
#         # 替换原始的全连接层
#         model.fc = nn.Sequential(
#             nn.Linear(num_features, 4096),
#             nn.ReLU(),
#             nn.BatchNorm1d(4096),
#             nn.Dropout(0.4),
#             nn.Linear(4096, 2048),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(2048, 2)
#         )

    # def forward(self, x):
    #     return self.resnet101(x)

    # def forward_with_probs(self, x):
    #     logits = self.forward(x)
    #     probs = F.softmax(logits, dim=1)
    #     return logits, probs
    
# def ResNet101(in_channels, num_classes, use_se=True):
#     return CustomResNet101(in_channels=in_channels, num_classes=num_classes, use_se=use_se)


# if __name__ == "__main__":
#     model = ResNet101(in_channels=3, num_classes=2, use_se=True)