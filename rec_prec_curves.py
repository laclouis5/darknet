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

def plot_rec_prec_curve(data, label):
    plt.figure()
    for i, d in enumerate(data):
        dist = d[0, 1]
        recall = d[:, 6]
        precision = d[:, 7]
        best = d[np.argmax(d[:, 8], axis=0), :]
        (best_rec, best_pre, best_f1) = best[6], best[7], best[8]

        plt.plot(recall, precision, label="maxDist: {:.0%}".format(dist))

        best_curve_index = 3 if label == "maize" else 2
        if i == best_curve_index:
            plt.annotate("{:.2%}".format(best_f1), (best_rec, best_pre))

    # BEAN
    if label == "bean":
        # rec, prec = 0.7979, 0.9146
        rec, prec = 0.7766, 0.9241
    else:
        # rec, prec = 0.8065, 0.8475
        rec, prec = 0.7661, 0.8716
    f1 = f1_score(rec, prec)
    plt.annotate("{:.2%}".format(f1), (rec + 0.005, prec + 0.005))
    plt.plot([rec], [prec], "rx", label="without aggregation")

    percents = np.linspace(0, 1, 11)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xticks(percents, ["{:.0%}".format(p) for p in percents])
    plt.yticks(percents, ["{:.0%}".format(p) for p in percents])
    plt.legend(loc="lower left")
    plt.title("Precision-recall curve as a function of $minDets$ and $maxDist$")
    plt.xlim([0.4, 1])  # 0.4
    plt.ylim([0.5, 1])  # 0.6
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
    plt.title("F1-score as a function of minDets and maxDist")
    plt.show()

def main():
    label = "maize"
    file = f"save/aggr_grid_search_{label}_2.csv"
    data = read_csv_file(file)
    data = add_info(data, 94 if label == "bean" else 124)
    data_by_dist = split_by_dist(data, 6)
    plot_rec_prec_curve(data_by_dist, label)

    return 0

if __name__ == "__main__":
    main()
