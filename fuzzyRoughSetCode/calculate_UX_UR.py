import pandas as pd
from fuzzyRoughSetCode import clustering, TFRS
from fuzzyRoughSetCode.TFRS import I_TFRS

def calculate_UX_UR(filename, decision_col, col_to_drop):
    table = pd.read_csv('../data/' + filename)

    for column in table:
        if ((column != decision_col) and (column != col_to_drop)):
            table[column] = clustering.cluster(table[column]).astype(str)

    table = table.drop(columns=[col_to_drop])
    # table.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\weather_clustered.csv',
    #                                       index=None, header=True)

    tfrs = I_TFRS()
    tfrs.fit(table[:100], decision_col)
    reduct = tfrs.reduct_attr

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
    for column in table:
        if column in col_name_reduct:
            U_R[column] = {}

            unique_values = table[column].unique()
            for value in unique_values:
                U_R[column][value] = {}

            index = 0
            for value in table[column]:
                U_R[column][value][len(U_R[column][value])] = index
                index = index + 1

    return col_name_reduct, U_X, U_R