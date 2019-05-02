#利用sklearn的库进行knn算法的建立与预测
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
knn.fit(iris.data, iris.target)

waitForPredict = [[6,2,9,0.3]]
predictLabel = knn.predict(waitForPredict)

# print(iris.data)#数据值Data
# print(iris.target)#数据标签Label
print (predictLabel)






















# #encoding:utf-8
# import csv
# import random
# import math
# import operator
 
# def loadDataset(filename, split, trainingSet=[] , testSet=[]):
# 	with open(filename, 'rb') as csvfile:
# 	    lines = csv.reader(csvfile)
# 	    dataset = list(lines)
# 	    for x in range(len(dataset)-1):
# 	        for y in range(4):
# 	            dataset[x][y] = float(dataset[x][y])
# 	        if random.random() < split:
# 	            trainingSet.append(dataset[x])
# 	        else:
# 	            testSet.append(dataset[x])
 
 
# def euclideanDistance(instance1, instance2, length):
# 	distance = 0
# 	for x in range(length):
# 		distance += pow((instance1[x] - instance2[x]), 2)
# 	return math.sqrt(distance)
 
# def getNeighbors(trainingSet, testInstance, k):
# 	distances = []
# 	length = len(testInstance)-1
# 	for x in range(len(trainingSet)):
# 		dist = euclideanDistance(testInstance, trainingSet[x], length)
# 		distances.append((trainingSet[x], dist))
# 	distances.sort(key=operator.itemgetter(1))
# 	neighbors = []
# 	for x in range(k):
# 		neighbors.append(distances[x][0])
# 	return neighbors
 
# def getResponse(neighbors):
# 	classVotes = {}
# 	for x in range(len(neighbors)):
# 		response = neighbors[x][-1]
# 		if response in classVotes:
# 			classVotes[response] += 1
# 		else:
# 			classVotes[response] = 1
# 	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
# 	return sortedVotes[0][0]
 
# def getAccuracy(testSet, predictions):
# 	correct = 0
# 	for x in range(len(testSet)):
# 		if testSet[x][-1] == predictions[x]:
# 			correct += 1
# 	return (correct/float(len(testSet))) * 100.0
	
# def main():
# 	# prepare data
# 	trainingSet=[]
# 	testSet=[]
# 	split = 0.67
# 	loadDataset('iris.data', split, trainingSet, testSet)
# 	print 'Train set: ' + repr(len(trainingSet))
# 	print 'Test set: ' + repr(len(testSet))
# 	# generate predictions
# 	predictions=[]
# 	k = 3
# 	for x in range(len(testSet)):
# 		neighbors = getNeighbors(trainingSet, testSet[x], k)
# 		result = getResponse(neighbors)
# 		predictions.append(result)
# 		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
# 	accuracy = getAccuracy(testSet, predictions)
# 	print('Accuracy: ' + repr(accuracy) + '%')
	
# main()


























# ##############################################################################
# ##############################################################################
# # from sklearn import datasets
# # from sklearn.model_selection import train_test_split
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn import metrics


# # #Load dataset
# # wine = datasets.load_wine()

# # # Split dataset into training set and test set
# # X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test


# # #Create KNN Classifier
# # knn = KNeighborsClassifier(n_neighbors=5)


# # #Train the model using the training sets
# # knn.fit(X_train, y_train)


# # #Predict the response for test dataset
# # y_pred = knn.predict(X_test)

# # # accuracy = metrics.accuracy_score(y_test, y_pred)

# # # # Model Accuracy, how often is the classifier correct?
# # # print(accuracy)

# # print(y_pred)










