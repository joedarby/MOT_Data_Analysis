import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                                   'VehicleClass':np.uint8,
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
df['MakeCode'] = df['Make'].cat.codes

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

features = df.as_matrix(['Mileage', 'VehicleAge', 'MakeCode'])
targets = df.as_matrix(['TestResult']).ravel()
vehicleAges = df.as_matrix(['VehicleAge'])
mileages = df.as_matrix(['Mileage'])
makes = df.as_matrix(['Make'])

codeMap = dict()
makeGroups = df.groupby(['Make', 'MakeCode'])
for group, result in makeGroups:
    codeMap[group[0]] = group[1]

#print(features)
#print(targets)

#output = plt.scatter(makes,targets.transpose())
#plt.show()

logistic.doLogReg(features,targets, codeMap)
#KN.doKN(features,targets)

del df


