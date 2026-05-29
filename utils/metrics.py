import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    Multi-class node classification metrics.
    :param predicts: Tensor
        - recommended shape: (num_samples, num_classes) probabilities or logits
        - if shape is (num_samples,), will be treated as binary probs (backward compat)
    :param labels: Tensor, shape (num_samples,)
    """
    y_true = labels.cpu().numpy()

    p = predicts.detach().cpu()
    if p.ndim == 2:
        # if logits are passed in, convert to probs then argmax
        # (softmax is monotonic for argmax, but we also want probs sometimes)
        y_pred = torch.argmax(p, dim=1).numpy()
    else:
        # fallback: binary threshold
        y_pred = (p.numpy() >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
