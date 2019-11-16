from fuzzyRoughSetCode import calculate_UX_UR

filename = 'weather_features.csv'
decision_col = 'weather_main'
col_to_drop = 'dt_iso'

col_name_reduct, U_X, U_R = calculate_UX_UR.calculate_UX_UR(filename, decision_col, col_to_drop)
# print(col_name_reduct)
# print(U_R)