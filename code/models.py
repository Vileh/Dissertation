import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import linalg
from torchvision import datasets

from training import get_transforms

# EfficientNet-B3 with Crossentropy Loss
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, fl_feature_scaler=2):
        super(EfficientNetModel, self).__init__()
        self.effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.effnet.classifier = nn.Identity()
        num_features = 1536

        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_features, fl_feature_scaler * num_features)
        self.bn2 = nn.BatchNorm1d(fl_feature_scaler * num_features)
        self.classifier = nn.Linear(fl_feature_scaler * num_features, num_classes)

    def forward(self, images):
        features = self.effnet(images)
        features = self.bn(features)
        features = self.dropout(features)
        features = self.fc(features)
        embeddings = self.bn2(features)
        logits = self.classifier(embeddings)
        return logits
    
# ArcFace Class
class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s

        return cos

# EfficientNet-B3 with ArcFace Loss
class ArcFaceEN(nn.Module):
    def __init__(self, num_classes, fl_feature_scaler=2, arcface_s=64, arcface_m=0.5):
        super(ArcFaceEN, self).__init__()
        self.effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.effnet.classifier = nn.Identity()
        num_features = 1536

        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_features, fl_feature_scaler * num_features)
        self.bn2 = nn.BatchNorm1d(fl_feature_scaler * num_features)
        self.arcface = ArcFace(fl_feature_scaler * num_features, num_classes, s=arcface_s, m=arcface_m)

    def forward(self, images, labels=None):
        features = self.effnet(images)
        features = self.bn(features)
        features = self.dropout(features)
        features = self.fc(features)
        embeddings = self.bn2(features)

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits
        else:
            return embeddings
        
# CustomImageFolder Class
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, label_map, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.label_dict = label_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = self.label_dict[self.classes[target]]  # Map the label string to integer index
        return sample, label
    
def load_model_and_data(model_name, loss, split, label_map, batch_size=64):
    _, transform_clean = get_transforms()

    # Load the datasets
    test_dataset = CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/test', label_map=label_map[split], transform=transform_clean)
    val_dataset = CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/val', label_map=label_map[split], transform=transform_clean)
    clean_train_dataset = CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/train', label_map=label_map[split], transform=transform_clean)

    # Create data loaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=False)

    # Get number of classes
    num_classes = len(set(clean_train_dataset.classes))

    if loss == 'crossentropy':
        model = EfficientNetModel(num_classes)
        weights_path = f'../results/weights/{model_name}'
    elif loss == 'arcface':
        model = ArcFaceEN(num_classes)
        weights_path = f'../results/weights/{model_name}'
    else:
        raise ValueError("Invalid loss type. Choose 'crossentropy' or 'arcface'.")

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    return {
        'model': model,
        'test_loader': test_loader,
        'val_loader': val_loader,
        'clean_train_loader': clean_train_loader,
        'num_classes': num_classes
    }