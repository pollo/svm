from svm import SVM

import pprint
import pickle
import random
import pylab
import numpy
from cvxopt.base import matrix

def generate_2d_points(n_points, label,
                       x_mean, x_var,
                       y_mean, y_var):
    return [[(random.normalvariate(x_mean, x_var),
             random.normalvariate(y_mean, y_var)),
             label] for i in range(n_points)]

def store_points(points, filename):
    with open(filename, "w") as outfile:
        pickle.dump(points, outfile)

def load_points(filename):
    with open(filename) as infile:
        l = pickle.load(infile)
    return l

def print_data(points):
    for point in points:
        if point[1] == 1:
            pylab.plot(point[0][0],
                       point[0][1],
                       'bo')
        else:
            pylab.plot(point[0][0],
                       point[0][1],
                       'ro')

def print_boundaries(clf):
    xrange = numpy.arange(-5,5,0.1)
    yrange = numpy.arange(-5,5,0.1)

    grid = matrix([[clf.indicator((x,y)) for y in yrange]
                   for x in xrange])

    pylab.contour(xrange, yrange, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths=(1, 3, 1))

def print_classification(clf):
    xrange = numpy.arange(-5,5,0.5)
    yrange = numpy.arange(-5,5,0.5)

    points = [(x,y) for x in xrange for y in yrange]

    for point in points:
        if clf.predict(point)==1:
            pylab.plot(point[0],
                       point[1],
                       'go')
        else:
            pylab.plot(point[0],
                       point[1],
                       'mo')

if __name__ == "__main__":
    generate_new = False
    if generate_new:
        positive_points = generate_2d_points(5, 1, -1.5, 1, 0.5, 1)
        positive_points += generate_2d_points(5, 1, 1.5, 1, 0.5, 1)
        negative_points = generate_2d_points(10, -1, -0.5, 0.5, -0.5, 0.5)
        data = negative_points+positive_points
        random.shuffle(data)
        store_points(data, "points.txt")
    else:
        data = load_points("points.txt")

    print_data(data)

    clf = SVM('polynomial', with_slack=False, degree=2)
    clf.train(data)

    print_boundaries(clf)

    print_classification(clf)

    pylab.show()
