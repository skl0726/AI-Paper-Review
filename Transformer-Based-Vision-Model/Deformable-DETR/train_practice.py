import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import time

from utils import nested_tensor_from_tensor_list
from deformable_detr import build


# basic args for test
class Args:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    num_feature_levels = 4
    dec_n_points = 4
    last_height = 32
    last_width = 32
    num_queries = 100
    aux_loss = False
    cls_loss_coef = 1.0
    bbox_loss_coef = 1.0
    giou_loss_coef = 1.0
    focal_alpha = 0.25
    position_embedding = 'learned'
    lr_backbone = 1e-5
    backbone = 'resnet50'
    dilation = False
    set_cost_class = 1.0  # cost_class of HungarianMatcher
    set_cost_bbox = 1.0   # cost_bbox of HungarianMatcher
    set_cost_giou = 1.0   # cost_giou of HungarianMatcher
    num_classes = 6


class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target
    

# generate object related to the model
args = Args()
print("device:", args.device)
model, criterion, postprocessors = build(args)
model.to(args.device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# dummy data, dataset and dataloader
images = torch.rand(10, 3, 256, 256)
targets = [{'boxes': torch.rand(5, 4), 'labels': torch.randint(0, args.num_classes, (5,))} for _ in range(10)]

dataset = CustomDataset(images, targets)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return torch.stack(images, dim=0), targets

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


# train loop
model.train()
num_epochs = 10
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    for batch in dataloader:
        images, targets = batch
        images = nested_tensor_from_tensor_list(images).to(args.device)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}, Duration: {epoch_duration:.2f} seconds")

end_time = time.time()
total_duration = end_time - start_time
total_duration /= 60
print(f"Training completed in {total_duration:.2f} minutes")


# save model parameter
model_save_path = 'deformable_detr_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")