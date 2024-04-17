import numpy as np

class MyGaussianNBClassifier():
    def __init__(self, priors=None):
        self.priors = priors  #p(y)
        self.eps = 1e-9
        
    def fit(self, X, y):
        self.num_ex, self.num_feat = X.shape #находим количество выборок и объектов
        self.num_classes = len(np.unique(y)) #количество классов
        self.mean = {} #мат ожидание
        self.variance = {} #дисперсия
        flag = 1
        self.class_priors = {}
        if self.priors == None:
            flag = 0   
        
        for classes in range(self.num_classes):
            #print(classes)
            X_c = X[y == classes]
            self.mean[str(classes)] = np.mean(X_c, axis = 0)
            self.variance[str(classes)] = np.var(X_c, axis = 0)
            if flag == 0:
                self.class_priors[str(classes)] = X_c.shape[0]/X.shape[0]
            else:
                self.class_priors[str(classes)] = self.priors[classes]
                        
    def probability_function(self, X, m, d): #считаем p(x_i|y)
        constant = -self.num_feat/2 * np.log(2*np.pi) - 0.5*np.sum(np.log(d + self.eps))
        p = 0.5*np.sum(np.power(X - m, 2)/(d + self.eps), 1)
        return np.exp(constant - p)
        

    def predict_proba(self, X):
        p = np.zeros((len(X), self.num_classes))
        for classes in range(self.num_classes):
            prior = self.class_priors[str(classes)]
            p_c = self.probability_function(X, self.mean[str(classes)], self.variance[str(classes)])
            p[:,classes] = p_c*prior
            p_x = p.sum(axis=1).reshape(-1, 1)
        return p/p_x
    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, 1)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)