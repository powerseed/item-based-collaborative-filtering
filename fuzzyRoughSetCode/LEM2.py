import copy

import numpy as np

from fuzzyRoughSetCode.calculate_UX_UR import calculate_UX_UR
from fuzzyRoughSetCode.calculate_pos_neg import calculate_upper_lower
import pandas as pd

# filename = 'breast_cancer.csv'
# decision_col = 'diagnosis'
# col_to_drop = 'id'

filename = 'breast_cancer.csv'
decision_col = 'diagnosis'
col_to_drop = 'id'

table = pd.read_csv('../data/' + filename)
table = table.drop(columns=[col_to_drop])

col_name_reduct, U_X, U_R = calculate_UX_UR(table, decision_col)
allupper_rules, alllower_rules, allupper_records, alllower_records, allaR, allBnd = calculate_upper_lower(U_X, U_R)
#
# qweqwewq = copy.deepcopy(alllower_records)
# qweqwewq = qweqwewq["M"]
# qweqwewq = list(qweqwewq)
# qweqwewq.sort()
# print("U_X: ", U_X)
# print("col_name_reduct: ", col_name_reduct)
# print("alllower_rules: ", alllower_rules)
# print("alllower_records: ", qweqwewq)

def get_all_t():
    all_t = {}
    for condition in col_name_reduct:
        all_t[condition] = {}
        unique_values = table[condition].unique()
        for value in unique_values:
            all_t[condition][value] = set()
    return all_t


all_t = get_all_t()


def populate_all_t():
    for index, row in table.iterrows():
        for condition in col_name_reduct:
            all_t[condition][row[condition]].add(index)


populate_all_t()


def calculate_all_relevant_T_G(G):
    all_relevant_T_G = set()

    for condition in all_t:
        for value in all_t[condition]:
            intersection = all_t[condition][value].intersection(G)
            if len(intersection) != 0:
                t = [condition, value]
                all_relevant_T_G.add(tuple(t))

    return all_relevant_T_G


def get_all_item_in_T(T):
    all_set_t = []
    for t in T:
        all_set_t.append(all_t[t[0]][t[1]])

    all_items_in_T = set()

    if len(all_set_t) != 0:
        all_items_in_T = all_set_t[0]
        for t in all_set_t:
            all_items_in_T = all_items_in_T.intersection(t)

    return all_items_in_T


def get_all_item_in_hollow_T(hollow_T):
    all_items_in_hollow_T = set()

    for T in hollow_T:
        all_items_in_hollow_T = all_items_in_hollow_T.union(get_all_item_in_T(T))

    return all_items_in_hollow_T


def delete_t_from_T(T, t):
    for pair in T:
        if pair[0] == t[0] and pair[1] == t[1]:
            T.remove(t)
            break


def equal_two_T(T_1, T_2):
    is_equal = True
    list_T_1 = list(T_1)
    list_T_1.sort()
    list_T_2 = list(T_2)
    list_T_2.sort()
    if len(T_1) != len(T_2):
        is_equal = False
    else:
        index = 0
        while index < len(list_T_1):
            if list_T_1[index][0] != list_T_2[index][0] or list_T_1[index][1] != list_T_2[index][1]:
                is_equal = False
            index = index + 1

    return is_equal


def delete_T_from_hollow_T(hollow_T, T_to_delete):
    copy_hollow_T = copy.deepcopy(hollow_T)
    the_T_to_delete = None

    for T in copy_hollow_T:
        if equal_two_T(T, T_to_delete):
            the_T_to_delete = T
            break

    copy_hollow_T.remove(the_T_to_delete)
    return copy_hollow_T

def select_a_t(all_relevant_T_G, G):
    list_all_relevant_T_G = list(all_relevant_T_G)
    list_cardinality = []

    index = 0
    while index < len(list_all_relevant_T_G):
        t = list_all_relevant_T_G[index]
        set_t = all_t[t[0]][t[1]]
        t_intersect_G = set_t.intersection(G)
        cardinality = len(t_intersect_G)
        list_cardinality.append(cardinality)
        index = index + 1

    max_cardinality = max(list_cardinality)
    indices_max_cardinality = []
    index = 0
    while index < len(list_cardinality):
        if list_cardinality[index] == max_cardinality:
            indices_max_cardinality.append(index)
        index = index + 1

    if len(indices_max_cardinality) == 1:
        return list_all_relevant_T_G[indices_max_cardinality[0]]
    else:
        ts_with_same_cardinality = []
        for index in indices_max_cardinality:
            ts_with_same_cardinality.append(list_all_relevant_T_G[index])

        cardinalities_t = []
        for t in ts_with_same_cardinality:
            cardinality_t = len(all_t[t[0]][t[1]])
            cardinalities_t.append(cardinality_t)

        min_cardinality = min(cardinalities_t)
        indices_min_cardinality = []
        index = 0
        while index < len(cardinalities_t):
            if cardinalities_t[index] == min_cardinality:
                indices_min_cardinality.append(index)
            index = index + 1

        if len(indices_min_cardinality) == 1:
            return ts_with_same_cardinality[indices_min_cardinality[0]]
        else:
            return list_all_relevant_T_G[0]

def lem2(lower_or_upper_set):
    conclusion = {}
    for decision in lower_or_upper_set:
        conclusion[decision] = []
        B = alllower_records[decision]
        G = copy.deepcopy(B)
        Covering_hollow_T = set()

        count = 0
        while (len(G) != 0):
            condition_T = set()
            all_relevant_T_G = calculate_all_relevant_T_G(G)  # set contains t
            # print("all_relevant_T_G: ", all_relevant_T_G)

            T_belongs_to_B = get_all_item_in_T(condition_T).issubset(B)

            while len(all_relevant_T_G) != 0 and (len(condition_T) == 0 or (not T_belongs_to_B)):  # while T = Ø or [T] Õ/ B
                t = select_a_t(all_relevant_T_G, G)
                all_relevant_T_G.remove(t)
                # print("condition_T 1: ", condition_T)
                # print("t 1: ", t)
                temp = set()
                temp.add(tuple(t))
                condition_T = condition_T.union(temp)
                G = G.intersection(all_t[t[0]][t[1]])  # G := [t] « G;
                all_relevant_T_G = calculate_all_relevant_T_G(G)
                all_relevant_T_G = all_relevant_T_G.difference(condition_T)  # T(G) := T(G) – T;
                T_belongs_to_B = get_all_item_in_T(condition_T).issubset(B)
            # end inner while

            # if len(condition_T) == 1:
            #     print("condition_T: ", condition_T, " count: ", count)
            #     break
            # for each t in T do
            set_t_to_delete_from_condition_T = []
            for t in condition_T:
                # print("condition_T 1: ", condition_T)
                copy_T = copy.deepcopy(condition_T)
                # print("copy 1: ", copy_T)
                # print("t 1: ", t)
                delete_t_from_T(copy_T, t)
                # print("copy 2: ", copy_T)

                if copy_T != None and len(copy_T) != 0:
                    all_items_in_remove_t = get_all_item_in_T(copy_T)
                    # print("copy_T: ", copy_T)
                    # print("all_items_in_remove_t: ", all_items_in_remove_t)
                    # print("B: ", B)
                    if all_items_in_remove_t.issubset(B):
                        # print("true")
                        set_t_to_delete_from_condition_T.append(t)

            if len(set_t_to_delete_from_condition_T) != 0:
                # print("set_t_to_delete_from_condition_T: ", set_t_to_delete_from_condition_T)
                list_condition_T = list(condition_T)
                # print("list_condition_T: before", list_condition_T)
                for t in set_t_to_delete_from_condition_T:
                    index = 0
                    while index < len(list_condition_T):
                        if list_condition_T[index][0] == t[0] and list_condition_T[index][1] == t[1]:
                            list_condition_T.pop(index)
                        else:
                            index = index + 1
                condition_T = set(list_condition_T)
                # print("condition_T: after", condition_T)
            # end: for each t in T do

            # print("Covering_hollow_T, before union: ", Covering_hollow_T)
            # print("condition_T: ", condition_T)
            temp = set()
            temp.add(tuple(condition_T))
            Covering_hollow_T = Covering_hollow_T.union(temp)
            # print("Covering_hollow_T, after union: ", Covering_hollow_T)

            # print("B1: ", B)
            # print("get_all_item_in_hollow_T(Covering_hollow_T): ", get_all_item_in_hollow_T(Covering_hollow_T))
            G = B.difference(get_all_item_in_hollow_T(Covering_hollow_T))
            # print("G1: after", G)
            count = count + 1
        # outer while
        T_to_delete = []
        for T in Covering_hollow_T:
            # print("Covering_hollow_T: ", Covering_hollow_T)
            # print("T: ", T)
            hollowT_minus_T = delete_T_from_hollow_T(Covering_hollow_T, T)
            # print("hollowT_minus_T: ", hollowT_minus_T)

            all_item_in_hollowT_minus_T = get_all_item_in_hollow_T(hollowT_minus_T)
            list_all_item_in_hollowT_minus_T = list(all_item_in_hollowT_minus_T)
            list_all_item_in_hollowT_minus_T.sort()
            list_B = list(B)
            list_B.sort()

            # print("list_all_item_in_hollowT_minus_T: ", list_all_item_in_hollowT_minus_T)
            # print("list_B: ", list_B)
            if list_all_item_in_hollowT_minus_T == list_B:
                print("true 2")
                # print("Covering_hollow_T 1: ", Covering_hollow_T)
                T_to_delete.append(T)
                # print("Covering_hollow_T 2: ", Covering_hollow_T)

        # print("T_to_delete: ", T_to_delete)
        # print("Covering_hollow_T: before", Covering_hollow_T)
        for T in T_to_delete:
            Covering_hollow_T.remove(T)

        conclusion[decision].append(Covering_hollow_T)
    return conclusion

conclusion_lower = lem2(alllower_records)
print("Conclusion for lower bound: ")
for decision in conclusion_lower:
    print("The conditions that are bound to cause the decision of ", decision, " are: ")
    for T in conclusion_lower[decision][0]:
        print(T)

conclusion_upper = lem2(allupper_records)
print("Conclusion for upper bound: ")
for decision in conclusion_upper:
    print("The conditions that could cause the decision of ", decision, " are: ")
    for T in conclusion_upper[decision][0]:
        print(T)
