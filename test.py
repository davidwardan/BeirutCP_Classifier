import os

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from utils import processing
from config import Config
from metrics import metrics


def main():
    # define configuration
    config = Config()

    # load test data
    print("loading test data....")
    x_test = processing.load_data(config.in_dir + '/test/x_test.npy')
    y_test = processing.load_data(config.in_dir + '/test/y_test.npy')
    x_test = processing.norm_image(x_test)
    y_test = processing.to_categorical(y_test, config.num_classes)

    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # load saved model
    print("loading model....")
    if config.saved_model_dir is not "":
        model = keras.models.load_model(config.saved_model_dir)
    else:
        model = keras.saving.load_model('saved_models/SwinT')

    # check if the output directory exists
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # predict on a test set
    print("predicting on test set....")
    y_pred = model.predict(x_test)

    # get accuracy
    print("calculating accuracy....")
    accuracy = metrics.get_accuracy(y_pred, y_test)

    # get precision
    print("calculating precision....")
    precision = metrics.get_precision(y_pred, y_test)

    # get recall
    print("calculating recall....")
    recall = metrics.get_recall(y_pred, y_test)

    # get confusion matrix
    print("calculating confusion matrix....")
    cm_normalized = metrics.get_confusion_matrix(y_pred, y_test)
    cm = metrics.get_confusion_matrix(y_pred, y_test, normalized=False)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=config.labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(config.out_dir + "/confusion_matrix_norm.png")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(config.out_dir + "/confusion_matrix.png")

    # get m_score and normalized m_score
    print("calculating m_score....")
    m_score = metrics.get_mscore(y_pred, y_test)
    print("calculating normalized m_score....")
    norm_m_score = metrics.get_normscore(cm_normalized, num_classes=config.num_classes)

    # print all scores in summary
    print("accuracy is equal to {}".format(accuracy))
    print("precision is equal to {}".format(precision))
    print("recall is equal to {}".format(recall))
    print("m_score is equal to {}".format(m_score))
    print("normalized m_score is equal to {}".format(norm_m_score))

    # optional: save scores to a dictionary
    # scores = {
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "m_score": m_score,
    #     "norm_m_score": norm_m_score
    # }

    # optional: save dictionary to a csv file
    # import pandas as pd
    # df_scores = pd.DataFrame(scores)
    # df_scores.to_csv(config.out_dir + "/scores.csv")


if __name__ == "__main__":
    main()
