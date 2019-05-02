from sklearn.externals import joblib
import sklearn
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics  


def knn_classifier(train_x, train_y):  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)#传入数据
    return model  


def knn(modelPath):
    classifiers = knn_classifier

    model = classifiers(X_train,y_train)  
    predict = model.predict(X_test)  

    accuracy = metrics.accuracy_score(y_test, predict)  
    print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  

    #保存模型
    joblib.dump(model,modelPath)


    model = joblib.load(modelPath)
    predict = model.predict(X_test) 
    accuracy = metrics.accuracy_score(y_test, predict)  
    print ('accuracy: %.2f%%' % (100 * accuracy)  ) 