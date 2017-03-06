from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

class LogRegModel(object):

    def __init__(self, featureMatrix, targets, catIndices):
        self.featureMatrix = featureMatrix
        self.targets = targets
        self.enc = preprocessing.OneHotEncoder(sparse=False, categorical_features=catIndices)
        self.model, self.scaler = self.log_reg()

    def log_reg(self):
        # If catIndices is empty encoder has no effect
        self.enc.fit(self.featureMatrix)
        features_encoded = self.enc.transform(self.featureMatrix)
        features_scaled = preprocessing.scale(features_encoded)
        scaler = preprocessing.StandardScaler().fit(features_encoded)

        model = LogisticRegression(random_state=None, max_iter=1000)
        model.fit(features_scaled, self.targets)
        print(model.classes_)
        print(model)
        # print("Coefficients = ", model.coef_)
        print("Intercept = %s" % model.intercept_)

        return model, scaler

    def continuous_only_plots(self, features):
        if (len(features) == 2):
            ax, X, Y = build_3d_plot("%s and %s" % (features[0], features[1]))
            self.plot_3d_no_cat(ax, X, Y, 0)
        else:
            x_max = np.amax(self.featureMatrix)
            ax, X = build_2d_plot([features[0]], x_max)
            self.plot_2d_no_cat(ax, X, 0)
        plt.show()

    def plot_for_categories(self, features, catsToPlot, codeMap):
        if (len(features) == 3):
            ax, X, Y = build_3d_plot(features[-1])
            for category in catsToPlot:
                code = codeMap.get(category)
                self.plot_cat_3d(ax, X, Y, 0, code, category)
        else:
            x_max = np.amax(self.featureMatrix)
            ax, X = build_2d_plot(features, x_max)
            for category in catsToPlot:
                code = codeMap.get(category)
                self.plot_cat_2d(ax, X, 0, code, category)
        plt.show()

    def getSingleProbability(self, predictor, resultType):
        return self.model.predict_proba(self.scaler.transform(self.enc.transform([predictor])))[0][resultType]

    def plot_cat_3d(self, ax, X, Y, modelClass, catCode, makeName):
        Z = self.get_probabilities_with_cat(X, Y, catCode, modelClass)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.text(-30000, -2000, Z[0][0] - 0.02, makeName)

    def plot_3d_no_cat(self, ax, X, Y, modelClass):
        Z = self.get_probabilities_no_cat(X, Y, modelClass)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    def plot_cat_2d(self, ax, X, modelClass, catCode, makeName):
        Y = self.get_probabilities_with_cat(X, None, catCode, modelClass)
        ax.plot(X, Y, linewidth=0.5)
        ax.text(0, Y[0], makeName)

    def plot_2d_no_cat(self, ax, X, modelClass):
        Y = self.get_probabilities_no_cat(X, None, modelClass)
        ax.plot(X, Y, linewidth=0.5)

    @np.vectorize
    def get_probabilities_with_cat(self, X, Y, catCode, resultType):
        if Y is None:
            return self.model.predict_proba(self.scaler.transform(self.enc.transform([[X, catCode]])))[0][resultType]
        else:
            return self.model.predict_proba(self.scaler.transform(self.enc.transform([[X, Y, catCode]])))[0][resultType]

    @np.vectorize
    def get_probabilities_no_cat(self, X, Y, resultType):
        if Y is None:
            return self.model.predict_proba(self.scaler.transform([[X]]))[0][resultType]
        else:
            return self.model.predict_proba(self.scaler.transform([[X, Y]]))[0][resultType]

















def build_3d_plot(categorical):
    X = np.arange(0, 300000, 10000)
    Y = np.arange(0, 20000, 1000)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Customize the z axis.
    ax.set_zlim(0.1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Age (days)')
    ax.set_title("Pass Rate By %s" % categorical)
    return ax, X, Y

def build_2d_plot(features, x_max):
    X = np.arange(0, x_max, (x_max/60))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(0.1, 1)
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(features[0])
    ax.set_title("Pass Rate By %s" % features[-1])
    return ax, X







