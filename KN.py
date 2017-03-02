from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

def doKN(features, targets):
    model = KNeighborsClassifier()
    model.fit(features, targets)
    print(model.classes_)
    #print("Intercept = %s" % model.intercept_)
    predicted = model.predict(features)
    print(metrics.classification_report(targets,predicted))
    print(metrics.confusion_matrix(targets,predicted))
    print(model)

    passMatrix = np.zeros([10,10])
    prsMatrix = np.zeros([10, 10])
    failMatrix = np.zeros([10, 10])
    for i in range(0,10):
        for j in range(0,10):
            passMatrix[i][j] = model.predict_proba([[(i* 20000),(j*1000)]])[0][0]
            prsMatrix[i][j] = model.predict_proba([[(i * 20000), (j * 1000)]])[0][2]
            failMatrix[i][j] = model.predict_proba([[(i * 20000), (j * 1000)]])[0][1]
            #print("%s miles, %s days: %f" % (i, j, model.predict_proba([[i,j]])[0][0]))
    print("Pass\n", passMatrix)
    print("\nPRS\n", prsMatrix)
    print("\nFail\n", failMatrix)