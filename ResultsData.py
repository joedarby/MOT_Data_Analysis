import numpy as np
import pandas as pd
from LogRegModel import LogRegModel


class ResultsData(object):
    testResultsHeader = ["TestID", "VehicleID", "TestDate", "VehicleClass", "TestType", "TestResult", "Mileage",
                         "PostcodeArea", "Make", "Model", "Colour", "FuelType", "EngineCapacity", "DateOfManufacture"]

    def __init__(self, sourcePath):
        self.sourcePath = sourcePath
        self.df, self.targets = self.build_df()

    def build_df(self):
        df = pd.read_table(self.sourcePath,
                   sep="|", header=None, names=ResultsData.testResultsHeader,
                   dtype={'TestID':np.uint32,
                                   'VehicleID':np.uint32,
                                   'Mileage':np.int32,
                                   'VehicleClass':'category',
                                   'EngineCapacity':np.uint16,
                                   'FuelType':'category',
                                   'TestType':'category',
                                   'TestResult':'category',
                                   'PostcodeArea':'category',
                                   'Make':'category',
                                   'Model':'category',
                                   'Colour':'category'},
                   parse_dates=[2, 13])
        df = df[df.Mileage < 300000]
        df['TestResult'] = pd.Categorical(df['TestResult'], categories=["P", "F", "PRS", "ABA", "ABR", "R", 2, 1, 0])
        df['TestResult'] = df.TestResult.replace(to_replace=["P", "F", "PRS", "ABA", "ABR", "R"],
                                                 value=[0, 2, 1, 2, 2, 2])
        df['TestResult'] = df['TestResult'].astype(np.uint8)
        df['VehicleAge'] = df['TestDate'] - df['DateOfManufacture']
        df['VehicleAge'] = df['VehicleAge'].dt.days.astype(np.uint32)
        df = df[df.VehicleAge < 20000]
        targets = df.as_matrix(['TestResult']).ravel()
        return df, targets

    def add_coded_column(self, category):
        codedColumnName = "%sCode" % category
        self.df[codedColumnName] = self.df[category].cat.codes
        return codedColumnName

    def get_code(self, category, value):
        codeMap = self.get_code_map(category)
        return codeMap.get(value)

    def get_code_map(self, category):
        codeMap = dict()
        makeGroups = self.df.groupby([category, ("%sCode" % category)])
        for group, result in makeGroups:
            codeMap[group[0]] = group[1]
        return codeMap

    def plot_one_or_two_continuous(self, features):
        featureMatrix = self.df.as_matrix(features)
        model = LogRegModel(featureMatrix, self.targets, [])
        model.continuous_only_plots(features)


    def plot_with_categorical(self, features, catsToPlot):
        codedFeatures = features[:]
        codedFeatures[-1] = self.add_coded_column(features[-1])
        codeMap = self.get_code_map(features[-1])
        featureMatrix = self.df.as_matrix(codedFeatures)
        model = LogRegModel(featureMatrix, self.targets, [-1])
        model.plot_for_categories(features, catsToPlot, codeMap)


    def calculate_one_probability(self, features, predictor, resultType):
        codedPredictor = predictor[:]
        codedFeatures = features[:]
        categoricalIndices = []
        i = 0
        for feature in features:
            if (str(self.df[feature].dtype) == "category"):
                codedFeatures[i] = self.add_coded_column(feature)
                codedPredictor[i] = self.get_code(feature, predictor[i])
                categoricalIndices.append(i)
            i += 1
        codedFeatures = self.df.as_matrix(codedFeatures)
        model = LogRegModel(codedFeatures, self.targets, categoricalIndices)
        result = model.getSingleProbability(codedPredictor, resultType)
        for i in range(len(features)):
            print("%s: %s" % (features[i], predictor[i]))
        print("Pass rate = %.2f%%" % (result * 100))


