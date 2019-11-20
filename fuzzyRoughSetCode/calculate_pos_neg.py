def calculate_upper_lower(U_X, U_R):
    allupper = {}
    alllower = {}
    allaR = []
    allBnd = []

    for decision in U_X:
        upperap = []
        lowerap = []
        U_Xcon = set()
        for item1 in U_X[decision]:
            U_Xcon.add(item1)
        for result in U_R:
            U_Rcon = set()
            # if decision is a subset of result
            for item in U_R[result]:
                U_Rcon.add(item)
            if (U_Rcon.issubset(U_Xcon)):
                lowerap.append(result)
            # if decision contains element of result
            if (U_Rcon.isdisjoint(U_Xcon) == False):
                upperap.append(result)
        allupper[decision] = upperap
        alllower[decision] = lowerap
        # allaR.add(len(lowerap)/len(upperap)) #粗糙度
    # allBnd.append(lowerap - upperap)#bond

    return allupper, alllower, allaR, allBnd