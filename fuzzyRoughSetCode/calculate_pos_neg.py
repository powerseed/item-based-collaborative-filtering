def calculate_upper_lower(U_X, U_R):
    allupper_rules = {}
    allupper_records = {}

    alllower_rules = {}
    alllower_records = {}

    allaR = []
    allBnd = []

    for decision in U_X:
        lowerap = []
        upperap = []

        lowerap_records = set()
        upperap_records = set()

        U_Xcon = set(U_X[decision])
        for result in U_R:
            U_Rcon = set(U_R[result])
            # if decision is a subset of result
            if (U_Rcon.issubset(U_Xcon)):
                lowerap.append(result)
                lowerap_records = lowerap_records.union(U_Rcon)
            # if decision contains element of result
            if (U_Rcon.isdisjoint(U_Xcon) == False):
                upperap.append(result)
                upperap_records = upperap_records.union(U_Rcon)

        alllower_rules[decision] = lowerap
        alllower_records[decision] = lowerap_records

        allupper_rules[decision] = upperap
        allupper_records[decision] = upperap_records

        # allaR.add(len(lowerap)/len(upperap)) #粗糙度
    # allBnd.append(lowerap - upperap)#bond

    return allupper_rules, alllower_rules, allupper_records, alllower_records, allaR, allBnd