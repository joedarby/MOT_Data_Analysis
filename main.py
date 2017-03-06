import numpy as np
from ResultsData import ResultsData
import MakeAndModel

np.set_printoptions(linewidth=320)

data = ResultsData('../../Documents/Data/test_result_2013_snippet.txt')
data.generate_model(['Mileage', 'VehicleAge', 'Make', 'FuelType', 'PostcodeArea'])
data.calculate_one_probability([100000, 1200, 'BMW', 'P', 'B'], 0)
data.calculate_one_probability([150000, 1200, 'BMW', 'P', 'B'], 0)
data.calculate_one_probability([150000, 1200, 'BMW', 'E', 'B'], 0)


#data.plot_with_categorical(['Mileage', 'Make'], ['ASTON MARTIN', 'AUDI', 'ROVER'])
#data.plot_with_categorical(['VehicleAge', 'Make'], ['ASTON MARTIN', 'AUDI', 'ROVER'])
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


#output = plt.scatter(makes,targets.transpose())
#plt.show()

#KN.doKN(features,targets)

'''
