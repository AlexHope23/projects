import numpy as np
class KnnBruteClassifier(object):

    def __init__(self, n_neighbors, weights, metric='l2'):
        self.k = n_neighbors
        self.weights = weights
        self.metric = metric
        
     
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        self.classes = np.unique(y)

        
    def predict(self, x: np.array):
        predictions = [] 
        neighbors = self.kneighbors(x, self.k)[1]
        distances = self.kneighbors(x, self.k)[0]
        for i in range(len(neighbors)):
            votes = []
            for j in range(len(neighbors[0])):
                votes.append(self.y_train[neighbors[i, j]]) #до этого нашли ближайших соседей, можем найти, к каким классам они принадлежат
            if self.weights == 'uniform':
                ans = np.bincount(votes).argmax()
            if self.weights == 'distance':
                w = 1/distances[i]
                weighted_votes = np.bincount(votes, weights=w)
                ans = np.argmax(weighted_votes)
            predictions.append(ans)      
        return np.array(predictions)
    
    def predict_proba(self, x: np.array):
        probabilities = np.zeros((len(x), len(np.unique(self.y_train))))
        distances, neighbors = self.kneighbors(x, self.k)

        for i in range(len(neighbors)):
            votes = []
            for j in range(len(neighbors[0])):
                votes.append(self.y_train[neighbors[i, j]])
            class_counts = [0] * len(self.classes)
            for x in votes:
                class_counts[x] += 1
            
            if self.weights == 'uniform':
                probabilities[i] = (class_counts / self.k)
                
            elif self.weights == 'distance':
                weighted_votes = [0] * len(self.classes)
                j = 0
                for x in votes:
                    weighted_votes[x] += 1/distances[i,j]      
                    j +=1
                weighted_votes = np.array(weighted_votes)
                    
                   
                probabilities[i] = (weighted_votes / weighted_votes.sum())
        return np.array(probabilities)
    
    def kneighbors(self, x, n_neighbors): #возвращает в виде массива расстояния от x до n_neighbors ближайших соседей и их индексы
        ind = np.zeros((len(x), n_neighbors), dtype = int)
        distances = np.zeros((len(x), n_neighbors))
        for i in range(len(x)):
            d = []
            for j in range(len(self.X_train)):
                dist = np.linalg.norm(self.X_train[j] - x[i])
                d.append([dist,int(j)])
            d.sort()
            d = d[:self.k]
            distances[i] = [item[0] for item in d]
            ind[i] = [item[1] for item in d]
        return distances, ind
