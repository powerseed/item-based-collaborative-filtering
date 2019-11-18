import pandas as pd
from fuzzyRoughSetCode import clustering, TFRS
from fuzzyRoughSetCode.TFRS import I_TFRS

def calculate_UX_UR(filename, decision_col, col_to_drop):
    table = pd.read_csv('../data/' + filename)
    table = table.drop(columns=[col_to_drop])

    for column in table:
        if column != decision_col:
            table[column] = clustering.cluster(table[column]).astype(str)

    table.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\weather_clustered.csv',
                                          index=None, header=True)

    decision = table[decision_col].values
    conditions = table.drop(columns=[decision_col]).values

    tfrs = I_TFRS()
    tfrs.fit(conditions, decision)
    reduct = tfrs.reduct_attr
    reduct = sorted(reduct)

    col_name_reduct = []
    for index in reduct:
        col_name_reduct.append(table.columns.values[index])

    values_decision = table[decision_col].unique()
    U_X = {}
    for value in values_decision:
        U_X[value] = {}

    index = 0
    for value in table[decision_col]:
        U_X[value][len(U_X[value])] = index
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
            U_R[values_conditions] = {}

        U_R[values_conditions][len(U_R[values_conditions])] = index

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