from svm import SVM

import pprint
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

positive_points = generate_2d_points(5, 1, -1.5, 1, 0.5, 1)
positive_points += generate_2d_points(5, 1, 1.5, 1, 0.5, 1)
negative_points = generate_2d_points(10, -1, -0.5, 0.5, -0.5, 0.5)

data = negative_points+positive_points
random.shuffle(data)

clf = SVM('rbf',sigma=2)

pprint.pprint(data)
clf.train(data)
pprint.pprint(clf.support_vector)

xrange = numpy.arange(-10,10,0.1)
yrange = numpy.arange(-10,10,0.1)

grid = matrix([[clf.indicator((x,y)) for y in yrange]
               for x in xrange])

pylab.hold(True)


pylab.contour(xrange, yrange, grid,
              (-1.0, 0.0, 1.0),
              colors=('red', 'black', 'blue'),
              linewidths=(1, 3, 1))


pylab.plot([p[0][0] for p in positive_points],
           [p[0][1] for p in positive_points],
           'bo')

pylab.plot([p[0][0] for p in negative_points],
           [p[0][1] for p in negative_points],
           'ro')

pylab.show()
