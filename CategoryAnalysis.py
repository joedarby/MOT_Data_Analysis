import numpy as np
from collections import OrderedDict

'''This function takes a category name and a dataframe and outputs group averages to a csv,
as described in the categorical fields exploration section of my report'''

def summarise(df, category, minGroupSize):
    passes = dict()
    groups = df.groupby(category)
    for groupName, group in groups:
        total = group.TestID.count()
        if total > minGroupSize:
            passAndFail = group.groupby('TestResult')
            for result, resultgroup in passAndFail:
                count = resultgroup.TestID.count()
                percent = round(count / total * 100, 1)
                if result == 0:
                    passes[(groupName, total)] = percent

    passes = OrderedDict(sorted(passes.items(), key=lambda t: t[1]))

    filePath = '../../Documents/Data/category_' + category + '.csv'
    file = open(filePath, 'w')
    for record in passes.keys():
        line = "%s, %s, %s\n" % (record[0], record[1], passes.get(record))
        file.write(line)

    file.close()

    print(category)
    print(passes)
    print(len(passes))













