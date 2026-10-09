from typing import Tuple, List

import numpy as np
import torch

import matplotlib.pyplot as plt

from .dino import ConvNeXtV2, DINOHead
from .dataset import NormalizeTransform


def show_images(images: List[Tuple[np.ndarray, str]], num_images: int = 5, normalize: bool = True, cmap='gray'):
    transform = NormalizeTransform()

    _, axes = plt.subplots(1, num_images, figsize=(15, 3))
    if num_images == 1:
        axes = (axes, )

    for i, (image, title) in enumerate(images):
        img = transform(image).squeeze() if normalize else image
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.show()


def fast_confusion_matrix(y_true, y_pred, num_classes):
    # Flatten arrays (crucial if passing 2D/3D image masks directly)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Filter out pixels not in your standard class range (like ignore_index)
    mask = (y_true >= 0) & (y_true < num_classes)
    
    # Map 2D coordinates (true_class, pred_class) to a 1D index and count
    hist = np.bincount(
        num_classes * y_true[mask].astype(int) + y_pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return hist


def generate_evaluation_report(y_true, y_pred, labels=None, digits=3):
    """
    Lightning-fast unified evaluation report returning a formatted string.
    Includes Normalized Confusion Matrix, Classification Metrics, and IoU.
    """
    if labels is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
        str_labels = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(labels)
        str_labels = [str(l) for l in labels]

    # Get the N x N confusion matrix
    cm = fast_confusion_matrix(y_true, y_pred, num_classes)
    
    # Extract core components
    tp = np.diag(cm)
    support = cm.sum(axis=1)       # Row sums (True labels)
    pred_totals = cm.sum(axis=0)   # Column sums (Predicted labels)
    
    # Calculate Normalized Confusion Matrix
    cm_norm = np.divide(
        cm.astype(float),
        support[:, np.newaxis],
        out=np.zeros_like(cm, dtype=float),
        where=support[:, np.newaxis] != 0
    )

    # Calculate Vectorized Metrics
    precision = np.divide(tp, pred_totals, out=np.zeros_like(tp, dtype=float), where=pred_totals != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
    
    f1_denom = precision + recall
    f1 = np.divide(2 * precision * recall, f1_denom, out=np.zeros_like(tp, dtype=float), where=f1_denom != 0)
    
    union = support + pred_totals - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union != 0)

    # Build the Output String
    out = []
    
    # --- Part A: Normalized Confusion Matrix ---
    out.append("Confusion Matrix (Normalized):")
    
    # Dynamic header spacing based on label lengths
    label_widths = [max(len(l), 7) for l in str_labels]
    header_labels = " ".join([f"{l:>{w}}" for l, w in zip(str_labels, label_widths)])
    header = f"{'True / Pred':>15} | {header_labels}"
    
    out.append(header)
    out.append("-" * len(header))
    
    for i, row_label in enumerate(str_labels):
        row_str = " ".join([f"{val:>{w}.3f}" for val, w in zip(cm_norm[i], label_widths)])
        out.append(f"{row_label:>15} | {row_str}")
        
    out.append("-" * len(header))
    out.append("\n")

    # --- Part B: Unified Classification & IoU Report ---
    name_width = max([len(l) for l in str_labels] + [12]) # 12 accommodates 'weighted avg'
    
    head_fmt = f"{{:>{name_width}}}  {{:>9}}  {{:>9}}  {{:>9}}  {{:>9}}  {{:>9}}"
    row_fmt  = f"{{:>{name_width}}}  {{:>9.{digits}f}}  {{:>9.{digits}f}}  {{:>9.{digits}f}}  {{:>9.{digits}f}}  {{:>9}}"
    
    out.append("Segmentation & Classification Report:")
    out.append(head_fmt.format("", "precision", "recall", "f1-score", "iou", "support"))
    out.append("")
    
    # Per-class metrics
    for i, label in enumerate(str_labels):
        out.append(row_fmt.format(label, precision[i], recall[i], f1[i], iou[i], int(support[i])))
        
    out.append("")
    
    # Global metrics
    total_support = np.sum(support)
    accuracy = np.sum(tp) / total_support if total_support > 0 else 0.0
    
    # Accuracy row (only displays in the iou and support columns to match sklearn layout)
    out.append(f"{'accuracy':>{name_width}}  {'':>9}  {'':>9}  {'':>9}  {accuracy:>9.{digits}f}  {int(total_support):>9}")
    
    # Macro average (Mean IoU / mIoU is naturally calculated here)
    out.append(row_fmt.format("macro avg", np.mean(precision), np.mean(recall), np.mean(f1), np.mean(iou), int(total_support)))
    
    # Weighted average
    if total_support > 0:
        wp = np.average(precision, weights=support)
        wr = np.average(recall, weights=support)
        wf1 = np.average(f1, weights=support)
        wiou = np.average(iou, weights=support)
    else:
        wp = wr = wf1 = wiou = 0.0
        
    out.append(row_fmt.format("weighted avg", wp, wr, wf1, wiou, int(total_support)))
    
    return "\n".join(out), (cm_norm, precision, recall, f1, iou, support)


def load_backbone(weights_path: str, device = torch.device("cuda")) -> ConvNeXtV2:
    """
    Loads the ConvNeXtTiny backbone from the training checkpoint.
    """
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    backbone_state_dict = {}
    if 'student' in checkpoint:
        backbone_state_dict = checkpoint['student']
    else:
        # Fallback if the user passes a raw state dict
        backbone_state_dict = checkpoint['backbone']

    if not backbone_state_dict:
        raise ValueError("No 'backbone.' keys found in the checkpoint. Check the weight file structure.")

    model = ConvNeXtV2(in_chans=1)
    model.load_state_dict(backbone_state_dict)
    model.to(device)
    model.eval()

    return model


def load_model(weights_path: str, output_dim: int = 4096, device = torch.device("cuda")) -> Tuple[ConvNeXtV2, DINOHead, DINOHead]:
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    dino_weights = checkpoint['student_dino_head']
    ibot_weights = checkpoint['student_ibot_head']

    backbone = load_backbone(weights_path, device)

    dino_head = DINOHead(in_dim=backbone.embed_dim, out_dim=output_dim)
    ibot_head = DINOHead(in_dim=backbone.embed_dim, out_dim=output_dim)

    for head, weights in ((dino_head, dino_weights), (ibot_head, ibot_weights)):
        head.load_state_dict(weights)
        head.to(device)
        head.eval()

    return backbone, dino_head, ibot_head


def run_inference(model: ConvNeXtV2, tile: np.ndarray, device = torch.device("cuda"), normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    transform = NormalizeTransform()

    with torch.no_grad():
        input_tensor = tile
        if normalize:
            input_tensor = transform(input_tensor)

        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Forward pass returns class embedding and patch embeddings
        stages = model._inference(input_tensor)

    outputs = []
    for stage in stages:
        outputs.append(stage.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    hypercolumn = model.fusion(stages)
    outputs.append(hypercolumn.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    cls, patch = model._get_output(hypercolumn)

    return cls.squeeze().cpu().detach().numpy(), patch.squeeze().cpu().detach().numpy(), outputs


def run_inference_heads(model: ConvNeXtV2, dino_head: DINOHead, ibot_head: DINOHead, tile: np.ndarray, device = torch.device("cuda")):
    cls, patch, outputs = run_inference(model, tile, device)

    dino_output = dino_head(torch.from_numpy(cls).to(device))
    ibot_output = ibot_head(torch.from_numpy(patch).to(device))

    return dino_output.squeeze().cpu().detach().numpy(), ibot_output.squeeze().cpu().detach().numpy(), outputs