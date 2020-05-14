import os
import numpy as np
import matplotlib.pyplot as plt

def f1_score(a, b):
    return 2 / (1/a + 1/b)

def precision(tp, dets):
    return tp / dets

def recall(tp, gts):
    return tp / gts

def read_csv_file(file):
    """
    (minPoints, minDist, TP, FP, accuracy)
    """
    extension = os.path.splitext(file)[1]
    assert extension == ".csv",  f"Invalid file, should be '.csv' not '{extension}'."

    with open(file, "r") as f:
        content = f.readlines()

    content = [line.strip().split(",") for line in content]
    data = np.array(content, dtype=float)

    return data

def add_info(data, gts):
    """
    (minPoints, minDist, TP, FP, accuracy, nbDets, recall, precision, f1Score)
    """
    nbDets = data[:, 2] + data[:, 3]
    prec = precision(data[:, 2], nbDets)
    rec = recall(data[:, 2], gts)
    f1 = f1_score(prec, rec)

    output = np.concatenate((data,
        nbDets[:, np.newaxis],
        rec[:, np.newaxis], prec[:, np.newaxis], f1[:, np.newaxis]),
        axis=1)

    return output

def split_by_dist(data, nb_splits):
    return np.vsplit(data, nb_splits)

def plot_rec_prec_curve(data):
    plt.figure()
    for d in data:
        dist = d[0, 1]
        recall = d[:, 6]
        precision = d[:, 7]
        plt.plot(recall, precision, label="dist: {:.0%}".format(dist))

    percents = np.linspace(0, 1, 11)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xticks(percents, ["{:.0%}".format(p) for p in percents])
    plt.yticks(percents, ["{:.0%}".format(p) for p in percents])
    plt.legend()
    plt.title("Precision-recall curve as a function of minPoints and maxDist")
    plt.show()

def plot_f1_curve(data):
    plt.figure()
    for d in data:
        dist = d[0, 1]
        f1 = d[:, 8]
        nb_points = d[:, 0]
        plt.plot(nb_points, f1, label="dist: {:.0%}".format(dist))

    points = np.arange(1, 21, 2)
    percents = np.linspace(0.5, 1, 6)
    plt.xlabel("Min. Points")
    plt.ylabel("F1-score")
    plt.xticks(points)
    plt.yticks(percents, ["{:.0%}".format(p) for p in percents])
    plt.legend()
    plt.title("F1-score as a function of minPoints and maxDist")
    plt.show()

def main():
    file = "aggr_grid_search_bean.csv"
    data = read_csv_file(file)
    data = add_info(data, 94)
    data_by_dist = split_by_dist(data, 20)

    # plot_rec_prec_curve(data_by_dist[2::2])
    plot_f1_curve(data_by_dist[2::2])

    return 0

if __name__ == "__main__":
    main()
