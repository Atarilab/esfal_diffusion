from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import numpy as np

def evaluate_multiclass(model, dataloader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"]
            labels = batch["target"]

            outputs = model(inputs)

            all_labels.extend(labels)
            all_outputs.extend(outputs)
    
    # Multiclass accuracy
    all_outputs = torch.from_numpy(np.array(all_outputs))
    all_labels = torch.from_numpy(np.array(all_labels))
    all_labels_id = torch.argmax(all_labels, dim=-1).numpy()
    all_preds_id = torch.argmax(all_outputs, dim=-1).numpy()
    accuracy = accuracy_score(all_labels_id, all_preds_id)
    
    # ROC score
    all_outputs = torch.nn.functional.softmax(all_outputs, dim=-1)
    roc_auc = roc_auc_score(all_labels.numpy(), all_outputs.numpy())
    return accuracy, roc_auc

def evaluate_binary(model, dataloader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"]
            labels = batch["target"]

            outputs = model(inputs)

            all_labels.extend(labels.numpy())
            all_outputs.extend(outputs.numpy())
    
    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    # Convert multiclass labels to binary labels (1 for first class, 0 for others)
    binary_labels = (all_labels[:, 0] == 1).astype(int)
    
    # Apply softmax to the outputs
    all_outputs_softmax = torch.nn.functional.softmax(torch.from_numpy(all_outputs), dim=-1).numpy()
    
    # Get the probabilities for the first class
    binary_outputs = all_outputs_softmax[:, 0]
    
    # Compute binary accuracy
    binary_preds = (binary_outputs > 0.5).astype(int)
    accuracy = accuracy_score(binary_labels, binary_preds)
    
    # Compute ROC AUC score
    if all_outputs.shape[-1] == 1:
        roc_auc = roc_auc_score(binary_labels, all_outputs)
    else:
        roc_auc = roc_auc_score(binary_labels, binary_outputs)

    return accuracy, roc_auc