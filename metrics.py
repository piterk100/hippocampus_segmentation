import torch

def soft_dice(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    # prob, gt: (N,1,H,W)
    dims = (1,2,3)                   # sumuj po kanałach i przestrzeni
    inter = (prob * gt).sum(dim=dims)
    den   = prob.sum(dim=dims) + gt.sum(dim=dims)
    return ((2*inter + eps) / (den + eps)).mean()

def dice_fg_only(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice liczony tylko dla klatek z fg>0; jeśli brak fg, zwraca None.
    """
    if gt.sum() == 0:
        return torch.tensor(0.0, device=prob.device)
    return soft_dice(prob, gt, eps)

def soft_dice_global(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    'Stara' wersja Dice: globalnie po całym batchu (jak miałeś na początku).
    """
    inter = (prob * gt).sum()
    den   = prob.sum() + gt.sum()
    return (2*inter + eps) / (den + eps)

def precision_recall(prob: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-8):
    """
    Precision/Recall z twardym progiem.
    Zwraca: precision(float), recall(float), tp, fp, fn, tn (int)
    """
    pred = (prob > thr).float()
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    return precision, recall, tp, fp, fn, tn
