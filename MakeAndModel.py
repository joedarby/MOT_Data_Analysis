import numpy as np
from collections import OrderedDict

def summariseMakes(df):
    passes = dict()
    fails = OrderedDict()
    prs = OrderedDict()

    makeGroups = df.groupby('Make')
    for make, result in makeGroups:
        total = result.Make.count()
        if total > 100:
            resultGroups = result.groupby('TestResult')
            for group, resultgroup in resultGroups:
                #print("%s %s" % (make, group))
                count = resultgroup.Make.count()
                percent = count / total * 100
                #print("%s = %.1f %%" % (count, percent))
                if group == 0:
                    passes[make] = percent
                elif group == 1:
                    prs[make] = percent
                elif group == 2:
                    fails[make] = percent

    #passes = OrderedDict(sorted(passes.items(), key=lambda t: t[1]))
    fails = OrderedDict(sorted(fails.items(), key=lambda t: t[1]))
    prs = OrderedDict(sorted(prs.items(), key=lambda t: t[1]))

    #print(passes.get("ROVER"))
    #print(fails)
    #print(prs)

    print(type(passes))

    passesArray = np.array(passes.items(), np.dtype)

    print(type(passesArray))
    print(passesArray)












