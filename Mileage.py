import matplotlib.pyplot as plt
import numpy as np

def analyse_mileage(df):
    limit = 250000
    bins = list(range(0, limit, int(limit/40)))

    passes = df[df.TestResult == 0].as_matrix(['Mileage']).ravel()
    fails = df[df.TestResult == 1].as_matrix(['Mileage']).ravel()

    print ("By Mileage:")
    print ("Passes: ", summary_stats(passes))
    print ("Fails: ", summary_stats(fails))

    plt.hist(passes, bins, color='blue', label='Passes')
    plt.hist(fails, bins, color='red', label='Fails')
    plt.title("Histogram of passes and\n fails by mileage")
    plt.legend()
    plt.xlabel("Mileage")
    plt.show()

def summary_stats(array):
    return "Mean = ", int(np.mean(array)), " Median = ", int(np.median(array)), " StdDev = ", int(np.std(array))