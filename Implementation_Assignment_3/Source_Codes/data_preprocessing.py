"""
Dataset: pa3_train_reduced.csv
validation: pa3_valid_reduced.csv

assign labels +1 to number 3 and -1 to label 5.
"""

import csv
import numpy

#read training labels,features
def open_csv(filename):
    f = open(filename + ".csv", 'r')
    reader = csv.reader(f)
    labels = []
    features = []

    for i, row in enumerate(reader):
        if i<100:
            labels.append(float(row[0]))
            features.append([float(x) for x in row[1:]])

    features = numpy.array(features)
    labels = numpy.array(labels)

    labels[labels == 3.0] = 1
    labels[labels == 5.0] = -1

    return labels, features








