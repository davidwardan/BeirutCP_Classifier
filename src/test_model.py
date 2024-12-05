import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils import processing
from config import Config
from src.metrics import metrics


def main():
    # Define configuration
    config = Config()

    # Load test data
    print("Loading test data....")
    x_test = processing.load_data(config.in_dir + "/test/x_test.npy")
    y_test = processing.load_data(config.in_dir + "/test/y_test.npy")
    x_test = processing.norm_image(x_test)
    y_test = processing.to_categorical(y_test, config.num_classes)

    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # Convert test data to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.argmax(axis=1), dtype=torch.long)

    # Load saved model
    print("Loading model....")
    if config.saved_model_dir:
        model = torch.load(config.saved_model_dir)
    else:
        model = torch.load("saved_models/SwinT")

    model.eval()

    # Check if the output directory exists
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # Predict on the test set
    print("Predicting on test set....")
    with torch.no_grad():
        y_pred_logits = model(x_test_tensor)
        y_pred = torch.softmax(y_pred_logits, dim=1).cpu().numpy()

    # Calculate accuracy
    print("Calculating accuracy....")
    accuracy = metrics.get_accuracy(y_pred, y_test)

    # Calculate precision
    print("Calculating precision....")
    precision = metrics.get_precision(y_pred, y_test)

    # Calculate recall
    print("Calculating recall....")
    recall = metrics.get_recall(y_pred, y_test)

    # Calculate confusion matrix
    print("Calculating confusion matrix....")
    cm_normalized = metrics.get_confusion_matrix(y_pred, y_test)
    cm = metrics.get_confusion_matrix(y_pred, y_test, normalized=False)

    # Plot and save confusion matrices
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=config.labels
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(config.out_dir, "confusion_matrix_norm.png"))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(config.out_dir, "confusion_matrix.png"))

    # Calculate custom scores
    print("Calculating m_score....")
    m_score = metrics.get_mscore(y_pred, y_test)
    print("Calculating normalized m_score....")
    norm_m_score = metrics.get_normscore(cm_normalized, num_classes=config.num_classes)

    # Print all scores in summary
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"m_score: {m_score}")
    print(f"Normalized m_score: {norm_m_score}")

    # Optionally save scores to a dictionary
    # scores = {
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "m_score": m_score,
    #     "norm_m_score": norm_m_score
    # }

    # Optionally save dictionary to a CSV file
    # import pandas
