import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logistic
import KN
import MakeAndModel

np.set_printoptions(linewidth=320)

testResultsHeader = ["TestID", "VehicleID", "TestDate", "VehicleClass", "TestType", "TestResult", "Mileage",
                    "PostcodeArea", "Make", "Model", "Colour", "FuelType", "EngineCapacity", "DateOfManufacture"]
testItemDetailHeader = ["TestID", "RfR", "RfRType", "Lat", "Long", "Vert", "DMark"]

df = pd.read_table('../../Documents/Data/test_result_2013_snippet.txt',
                   sep="|", header=None, names=testResultsHeader,
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
df['TestResult'] = pd.Categorical(df['TestResult'], categories=["P", "F", "PRS", "ABA", "ABR", "R",2,  1, 0])
#print("P = %s" % len(testResults[testResults.TestResult == "P"]))
#df['TestResult'] = df.TestResult.replace(to_replace=["P", "F", "PRS", "ABA", "ABR", "R"], value=["P", "F", "PRS", "F", "F", "F"])
df['TestResult'] = df.TestResult.replace(to_replace=["P", "F", "PRS", "ABA", "ABR", "R"], value=[0,2,1,2,2,2])
df['TestResult'] = df['TestResult'].astype(np.uint8)

df['VehicleAge'] = df['TestDate'] - df['DateOfManufacture']
df['VehicleAge'] = df['VehicleAge'].dt.days.astype(np.uint32)
df = df[df.VehicleAge < 20000]

#MakeAndModel.summariseMakes(df)

#print(testResults['TestResult'].dtypes)
#print(df.head(10)['VehicleAge'])
#print(df.dtypes)
#print(testResults.memory_usage().sum())

#dfpass = df[df.TestResult == 0]
#df2 = df2[df2.Mileage > 40000]

#dfPRS = df[df.TestResult == 0]
#df3 = df3[df.Mileage < 60000]

#dfFail = df[df.TestResult == 0]

#df = pd.concat([df2,df3])

#print(dfpass.groupby(by=['Make']).count() > 10)
#print(dfPRS.groupby(by=['Make']).Make.count())
#print(dfFail.groupby(by=['Make']).Make.count())




def analyseByCategory(df, category, names):
    codedColumnName = "%sCode" % category
    df[codedColumnName] = df[category].cat.codes
    features = df.as_matrix(['Mileage', 'VehicleAge', codedColumnName])
    targets = df.as_matrix(['TestResult']).ravel()
    codeMap = codeDictionary(df, category)
    logistic.plotForCategories(category, names, features, targets, codeMap)

def oneResult(df, features, predictor, resultType):
    codedPredictor = predictor[:]
    codedFeatures = features[:]
    categoricalIndices = []
    i = 0
    for feature in features:
        if(str(df[feature].dtype) == "category"):
            codedColumnName = "%sCode" % feature
            df[codedColumnName] = df[feature].cat.codes
            codedFeatures[i] = codedColumnName
            codedPredictor[i] = getCode(df, feature, predictor[i])
            categoricalIndices.append(i)
        i += 1
    codedFeatures = df.as_matrix(codedFeatures)
    targets = df.as_matrix(['TestResult']).ravel()
    result = logistic.getSingleProbability(codedFeatures, targets, codedPredictor, categoricalIndices, resultType)
    for i in range(len(features)):
        print("%s: %s" % (features[i], predictor[i]))
    print("Pass rate = %.2f%%" % (result*100))

def getCode(df, category, value):
    codeMap = dict()
    makeGroups = df.groupby([category, ("%sCode" % category)])
    for group, result in makeGroups:
        codeMap[group[0]] = group[1]
    return codeMap.get(value)

#analyseByCategory(df, 'FuelType', ['E', 'P'])

oneResult(df, ['Mileage', 'VehicleAge', 'FuelType', 'Make'], [50000, 500, 'P', 'ROVER'], 0)


#print(features)
#print(targets)

#output = plt.scatter(makes,targets.transpose())
#plt.show()

#KN.doKN(features,targets)

del df


