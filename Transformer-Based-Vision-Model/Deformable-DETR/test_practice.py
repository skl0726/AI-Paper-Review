import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

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


# Load the saved model parameters
model_save_path = 'deformable_detr_model.pth'
model.load_state_dict(torch.load(model_save_path, map_location=args.device))
model.eval()
print(f"Model loaded from {model_save_path}")


# dummy test data, dataset and dataloader
test_images = torch.rand(5, 3, 256, 256)
test_targets = [{'boxes': torch.rand(3, 4), 'labels': torch.randint(0, args.num_classes, (3,))} for _ in range(5)]

test_dataset = CustomDataset(test_images, test_targets)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return torch.stack(images, dim=0), targets

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


# test loop
model.eval()

for batch in test_dataloader:
    images, targets = batch
    images = nested_tensor_from_tensor_list(images).to(args.device)
    targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        outputs = model(images)

    print("=====")
    print("Test - outputs['pred_logits'].shape: ", outputs['pred_logits'].shape)
    print("Test - max index of pred_logits: ", torch.max(outputs['pred_logits'], dim=-1)[1])
    print("Test - outputs['pred_boxes'].shape: ", outputs['pred_boxes'].shape)
