import numpy as np
import pandas as pd
from ResultsData import ResultsData
import CategoryAnalysis
import Mileage
import Age

np.set_printoptions(linewidth=320)

'''Generate a dataframe from the subset data file'''
data = ResultsData('../../Documents/Data/test_result_2013_subset.txt')


'''Code to produce report is shown below (delete '#' to activate code as necessary)'''


'''Code to analyse data by continuous variables (mileage and age)'''
#Mileage.analyse_mileage(data.df)
#Age.analyse_age(data.df)

'''Code to produce tables in the report which summarise the data by categorical features'''
#CategoryAnalysis.summarise(data.df, 'MakeModel', 100)
#CategoryAnalysis.summarise(data.df, 'FuelType', 100)
#CategoryAnalysis.summarise(data.df, 'Make', 100)
#CategoryAnalysis.summarise(data.df, 'VehicleClass', 100)
#CategoryAnalysis.summarise(data.df, 'Colour', 500)


'''Report model 1'''
#data.generate_model(['Mileage'])
#data.plot_continuous_only()

'''Report model 2'''
#data.generate_model(['VehicleAge'])
#data.plot_continuous_only()

'''Report model 3'''
#data.generate_model(['Mileage', 'VehicleAge'])
#data.plot_continuous_only()

'''Report model 4'''
#data.generate_model(['Mileage', 'VehicleClass'])
#data.plot_with_categorical(['1','2','3','4','5','7'])

'''Report model 5'''
#data.generate_model(['VehicleAge', 'Make'])
#data.plot_with_categorical(['ASTON MARTIN', 'PORSCHE', 'SKODA', 'RENAULT'])

'''Report model 6'''
#data.generate_model(['Mileage', 'VehicleAge', 'FuelType'])
#data.plot_with_categorical(['E', 'P'])




