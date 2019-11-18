import this

import calculate_UX_UR

filename = 'breast_cancer.csv'
decision_col = 'diagnosis'
col_to_drop = 'id'

# parameter:
#   filename: the name of the database csv file.
#   decision_col: the name of the decision attribute
#   col_to_drop: column that cannot be processed when doing the reduct, like "date".
#
# return:
#   col_name_reduct: the names of remaining columns after the reduct
#   U_X: elementary sets based on the decision attribute
#   U_R: elementary sets based on all conditional attributes
col_name_reduct, U_X, U_R = calculate_UX_UR.calculate_UX_UR(filename, decision_col, col_to_drop)
print(col_name_reduct)

def calculate_upper_lower(U_X, U_R):
    allupper = []
    alllower = []
    allaR = []
    allBnd = []
    for decision in U_X:
        upperap = []
        lowerap = []
        U_Xcon = set()
        for item1 in U_X[decision]:
            U_Xcon.add(U_X[decision][item1])
        for result in U_R:
            U_Rcon = set()
            # if decision is a subset of result
            for item in U_R[result]:
                U_Rcon.add(U_R[result][item])
            if (U_Rcon.issubset(U_Xcon)):
                lowerap.append(result)
            # if decision contains element of result
            if (U_Rcon.isdisjoint(U_Xcon) == False):
                upperap.append(result)
        allupper.append(decision)
        allupper.append(upperap)  # upper approximation
        alllower.append(decision)
        alllower.append(lowerap)  # lower approximation
        # allaR.add(len(lowerap)/len(upperap)) #粗糙度
    # allBnd.append(lowerap - upperap)#bond
    return allupper, alllower, allaR, allBnd

allupper, alllower, allaR, allBnd = calculate_upper_lower(U_X, U_R)

print(alllower)
print(allupper)