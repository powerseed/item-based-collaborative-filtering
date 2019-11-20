import pandas as pd
from fuzzyRoughSetCode import clustering, TFRS
from fuzzyRoughSetCode.TFRS import I_TFRS

# parameter:
#   filename: the name of the database csv file.
#   decision_col: the name of the decision attribute
#   col_to_drop: column that cannot be processed when doing the reduct, like "date".
#
# return:
#   col_name_reduct: the names of remaining columns after the reduct
#   U_X: elementary sets based on the decision attribute
#   U_R: elementary sets based on all conditional attributes
def calculate_UX_UR(table, decision_col):
    for column in table:
        if column != decision_col:
            table[column] = clustering.cluster(table[column]).astype(str)

    table.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\weather_clustered.csv',
                                          index=None, header=True)

    decision = table[decision_col].values
    conditions = table.drop(columns=[decision_col]).values

    tfrs = I_TFRS()
    tfrs.fit(conditions[:100], decision)
    reduct = tfrs.reduct_attr
    reduct = sorted(reduct)

    col_name_reduct = []
    for index in reduct:
        col_name_reduct.append(table.columns.values[index])

    values_decision = table[decision_col].unique()
    U_X = {}
    for value in values_decision:
        U_X[value] = []

    index = 0
    for value in table[decision_col]:
        U_X[value].append(index)
        index = index + 1

    U_R = {}
    for index, row in table.iterrows():
        values_conditions = ""

        index_col_name_reduct = 0
        for col_name in col_name_reduct:
            values_conditions = values_conditions + row[col_name]

            if index_col_name_reduct != len(col_name_reduct) - 1:
                values_conditions = values_conditions + ", "

            index_col_name_reduct = index_col_name_reduct + 1

        if not values_conditions in U_R:
            U_R[values_conditions] = []

        U_R[values_conditions].append(index)

    # for column in table:
    #     if column in col_name_reduct:
    #         U_R[column] = {}
    #
    #         unique_values = table[column].unique()
    #         for value in unique_values:
    #             U_R[column][value] = {}
    #
    #         index = 0
    #         for value in table[column]:
    #             U_R[column][value][len(U_R[column][value])] = index
    #             index = index + 1

    return col_name_reduct, U_X, U_R