from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def doLogReg(features, targets, codeMap):
    enc = preprocessing.OneHotEncoder(sparse=False, categorical_features=[2])
    enc.fit(features)
    features_encoded = enc.transform(features)
    features_scaled = preprocessing.scale(features_encoded)
    scaler = preprocessing.StandardScaler().fit(features_encoded)


    #output = plt.scatter(features_scaled.transpose()[0], targets.transpose())
    #plt.show()

    model = LogisticRegression(random_state=None, max_iter=1000)
    model.fit(features_scaled, targets)
    print(model.classes_)

    predicted = model.predict(features_scaled)
    print(metrics.classification_report(targets,predicted))
    print(metrics.confusion_matrix(targets,predicted))
    print(model)
    print("Coefficients = ", model.coef_)
    print("Intercept = %s" % model.intercept_)

    passMatrix = np.zeros([20, 10])
    #prsMatrix = np.zeros([20, 10])
    #failMatrix = np.zeros([20, 10])
    #print(model.predict_proba([[100000,3000]]))
    #for i in range(0,20):
        #for j in range(0,10):
            #passMatrix[i][j] = model.predict_proba(scaler.transform([[(i* 10000),(j*1000)]]))[0][0]
            #prsMatrix[i][j] = model.predict_proba(scaler.transform([[(i* 10000),(j*1000)]]))[0][1]
            #failMatrix[i][j] = model.predict_proba(scaler.transform([[(i * 10000), (j * 1000)]]))[0][2]
            #print("%s miles, %s days: %f" % (i, j, model.predict_proba([[i,j]])[0][0]))
    #print("Pass\n", passMatrix)
    #print("\nPRS\n", prsMatrix)
    #print("\nFail\n", failMatrix)

    X = np.arange(0,300000,10000)
    Y = np.arange(0,20000,1000)
    X, Y = np.meshgrid(X, Y)

    makeName = "ROVER"
    make = codeMap.get(makeName)

    plotClass(X, Y, model, scaler, 0, enc, make)
    #plotClass(X, Y, model, scaler, 1)
    plotClass(X, Y, model, scaler, 2, enc, make)

@np.vectorize
def getProbabilities(X,Y, model, scaler, modelClass, enc, make):
    return model.predict_proba(scaler.transform(enc.transform([[X, Y, make]])))[0][modelClass]


def plotClass(X,Y, model, scaler, modelClass, enc, make):
    Z = getProbabilities(X, Y, model, scaler, modelClass, enc, make)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0.1, 0.8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Age (days)")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()





