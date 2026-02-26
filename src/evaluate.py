from sklearn.metrics import accuracy_score

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return accuracy_score(targets, preds)
