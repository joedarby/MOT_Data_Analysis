import numpy as np
from ResultsData import ResultsData
import logistic
import MakeAndModel

np.set_printoptions(linewidth=320)

data = ResultsData('../../Documents/Data/test_result_2013_snippet.txt')
data.plot_with_categorical(['Mileage', 'Make'], ['ASTON MARTIN', 'AUDI', 'ROVER'])
data.plot_with_categorical(['VehicleAge', 'Make'], ['ASTON MARTIN', 'AUDI', 'ROVER'])
#data.plot_with_categorical(['Mileage', 'VehicleAge', 'Make'], ['ASTON MARTIN', 'FORD'] )
#data.plot_one_or_two_continuous(['VehicleAge'])
#data.plot_one_or_two_continuous(['Mileage', 'VehicleAge'])
#data.calculate_one_probability(['Mileage', 'VehicleAge', 'FuelType', 'Make'], [50000, 500, 'P', 'ROVER'], 0)
#data.calculate_one_probability(['Mileage', 'VehicleAge'], [50000, 500], 0)
del data


#testItemDetailHeader = ["TestID", "RfR", "RfRType", "Lat", "Long", "Vert", "DMark"]


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

'''
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

#oneResult(df, ['Mileage', 'VehicleAge', 'FuelType', 'Make'], [50000, 500, 'P', 'ROVER'], 0)


#print(features)
#print(targets)

#output = plt.scatter(makes,targets.transpose())
#plt.show()

#KN.doKN(features,targets)

'''
