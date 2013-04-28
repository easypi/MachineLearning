import numpy as np
import pylab as pl

def mean_normalization(X_set):
    m, n = X_set.shape


    for j in range(1,n):
        x_j = X_set[:,j]
        x_j_mean = np.mean(x_j)
        x_j_range = np.max(x_j) - np.min(x_j)

        x_j = (x_j - x_j_mean) / (x_j_range + np.spacing(1))
        X_set[:,j] = x_j

    return X_set

def normal_equation(X, y, learning_rate = 0):
    theta = np.linalg.pinv(X.T * X) * X.T * y
    return theta

def batch_gradient(X, y, learning_rate = 1e-2):
    epsilon = 1e-3 # converged if error is less than epsilon
    max_iter = 5e4 # max iterator number of loop

    m, n = X.shape
        
    theta = np.asmatrix( np.zeros([n,1]) )
    ite = 0
    abs_err = list()
    abs_err.append(1e5)

    while ite < max_iter and abs_err[-1] > epsilon: 
        for j in range(n):
            theta[j] = theta[j] + learning_rate * 1/m * ( (y - X * theta).T * X[:,j] )

        abs_err.append( np.linalg.norm( X * theta - y ) )
        ite += 1
        # print str(ite) + " abs_err: " + str(abs_err[-1])

    pl.figure()
    pl.title( 'batch_gradient: ite='+ str(ite) + "; abs_err= " + str(abs_err[-1]))
    pl.plot(range(ite), abs_err[1:ite+1])
    pl.savefig('batch_gradient')
    # pl.show()

    return theta

def stochastic_gradient(X, y, learning_rate = 1e-2):
    epsilon = 1e-3 # converged if error is less than epsilon
    max_iter = 1e3 # max iterator number of loop

    m, n = X.shape
    theta = np.asmatrix( np.zeros([n,1]) )
    ite = 0
    abs_err = list()
    abs_err.append(1e5)

    while ite < max_iter and abs_err[-1] > epsilon: 
        for i in range(m):
            for j in range(n):
                theta[j] = theta[j] + learning_rate * ( y[i,:] - X[i,:] * theta) * X[i,j]
       
        ite += 1
        abs_err.append( np.linalg.norm( X.dot(theta) - y ) )
        # print str(ite) + " abs_err: " + str(abs_err[-1])
    
    pl.figure()
    pl.title( 'stochastic_gradient: ite=' + str(ite) + ", abs_err=" + str(abs_err[-1]))
    pl.plot(range(ite), abs_err[1:ite+1])
    pl.savefig('stochastic_gradient')
    # pl.show()


    return theta
    
class LinearRegression(object):
    """Linear regression model:"""
    def __init__(self, feature_scaling = None):
        self.feature_scaling = feature_scaling
        self.theta = None

    def train(self, X_train, y_train, algorithm = normal_equation, learning_rate = 0.01):

        if X_train.shape[0] != y_train.shape[0]:
            print "different size of X and y in training set"
        
        m, n = X_train.shape

        # feature scaling
        if self.feature_scaling is not None:
            X_train = self.feature_scaling(X_train)

        # training parameter
        self.theta = algorithm(X_train, y_train, learning_rate)

        print self.theta
        
    def predict(self, X_test):
        # feature scaling
        if self.feature_scaling is not None:
            X_test= self.feature_scaling(X_test)

        y_test = self.theta.T * X_test
        return y_test



def main():
    datasets = np.genfromtxt('data2.txt', delimiter=',')
    X_train = np.matrix (datasets[:, :-1])
    m, n = X_train.shape
    X_train = np.c_[np.ones([m,1]), X_train]
    y_train = np.matrix (datasets[:, -1]).T

    lr = LinearRegression(feature_scaling=mean_normalization)
    lr.train(X_train, y_train, algorithm = batch_gradient)

    # pl.scatter(X_train[:,1], y_train, label='training data')
    # pl.plot(X_train[:,1], X_train * lr.theta, 'r' ,label='linear regression')
    # pl.legend()
    # pl.show()


    lr.train(X_train, y_train, algorithm = normal_equation)

    # pl.scatter(X_train[:,1], y_train, label='training data')
    # pl.plot(X_train[:,1], X_train * lr.theta, 'r' ,label='linear regression')
    # pl.legend()
    # pl.show()

    lr.train(X_train, y_train, algorithm = stochastic_gradient)

    # pl.scatter(X_train[:,1], y_train, label='training data')
    # pl.plot(X_train[:,1], X_train * lr.theta, 'r' ,label='linear regression')
    # pl.legend()
    # pl.show()



if __name__ == '__main__':
    main()