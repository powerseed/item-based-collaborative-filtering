import calculate_UX_UR

filename = 'weather_features.csv'
decision_col = 'weather_main'
col_to_drop = 'dt_iso'

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
def calculate_upper_lower(U_X,U_R):
    counter = 0
    allupper,alllower,allaR,allBnd = {}
    for decision in U_X:
        upperap,lowerap = {}
        for result in U_R:
            #if decision is a subset of result
            if (set(decision).issubset(set(result))):
                upperap.add(decision)
            #if decision contains element of result
            elif((set(decision)).isdisjoint(set(result))):
                lowerap.add(decision)
        allupper[counter] = frozenset(upperap)#upper approximation
        alllower[counter] = frozenset(lowerap)#lower approximation
        allaR[counter] = len(lowerap)/len(upperap) #粗糙度
        allBnd[counter] = lowerap - upperap#bond
        ++counter
    return allupper,alllower,allaR,allBnd

# print(col_name_reduct)
# print(U_R)