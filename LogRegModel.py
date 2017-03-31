from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

'''This class holds the logistic regression implementation, the functions
which builds plots of the models and the functions which make model predictions'''

class LogRegModel(object):

    def __init__(self, features, featureMatrix, targets, catIndices):
        self.features = features
        self.featureMatrix = featureMatrix
        self.targets = targets
        self.enc = preprocessing.OneHotEncoder(sparse=False, categorical_features=catIndices)
        self.model, self.scaler = self.log_reg()


    '''The log_reg function performs the encoding, scaling and model generation described
    in the implementation section of the report'''

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
        print("Intercept = %s" % model.intercept_)

        return model, scaler

    def continuous_only_plots(self):
        if (len(self.features) == 2):
            ax, X, Y = build_3d_plot("%s and %s" % (self.features[0], self.features[1]))
            self.plot_3d_no_cat(ax, X, Y, 0)
        else:
            x_max = np.amax(self.featureMatrix)
            ax, X = build_2d_plot([self.features[0]], x_max)
            self.plot_2d_no_cat(ax, X, 0)
        plt.show()

    def plot_for_categories(self, catsToPlot, codeMap):
        if (len(self.features) == 3):
            ax, X, Y = build_3d_plot(self.features[-1])
            for category in catsToPlot:
                code = codeMap.get(category)
                self.plot_cat_3d(ax, X, Y, 0, code, category)
        else:
            x_max = np.amax(self.featureMatrix)
            ax, X = build_2d_plot(self.features, x_max)
            for category in catsToPlot:
                code = codeMap.get(category)
                self.plot_cat_2d(ax, X, 0, code, category)
        plt.legend()
        plt.show()

    def getSingleProbability(self, predictor, resultType):
        return self.model.predict_proba(self.scaler.transform(self.enc.transform([predictor])))[0][resultType]

    def plot_cat_3d(self, ax, X, Y, modelClass, catCode, makeName):
        Z = self.get_probabilities_with_cat(self, X, Y, catCode, modelClass)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.text(-30000, -2000, Z[0][0] - 0.02, makeName)

    def plot_3d_no_cat(self, ax, X, Y, modelClass):
        Z = self.get_probabilities_no_cat(self, X, Y, modelClass)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    def plot_cat_2d(self, ax, X, modelClass, catCode, makeName):
        Y = self.get_probabilities_with_cat(self, X, None, catCode, modelClass)
        ax.plot(X, Y, linewidth=1, label=makeName)

    def plot_2d_no_cat(self, ax, X, modelClass):
        print(modelClass)
        Y = self.get_probabilities_no_cat(self, X, None, 0)
        ax.plot(X, Y, linewidth=1)

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
    X = np.arange(0, 250000, 10000)
    Y = np.arange(0, 12000, 1000)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Customize the z axis.
    ax.set_zlim(0.1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Age (days)')
    ax.set_title("Probability of MOT Pass by %s" % categorical)
    return ax, X, Y


def build_2d_plot(features, x_max):
    X = np.arange(0, x_max, (x_max/60))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(0.1, 1)
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(get_xlabel(features))
    ax.set_title("Probability of MOT Pass\n by %s and %s" % (features[0], features[1]))
    return ax, X


def get_xlabel(features):
    if (features[0] == 'VehicleAge'):
        return "%s (days)" % features[0]
    else:
        return features[0]





