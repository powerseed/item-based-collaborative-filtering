import copy

import numpy as np

from fuzzyRoughSetCode.calculate_UX_UR import calculate_UX_UR
from fuzzyRoughSetCode.calculate_pos_neg import calculate_upper_lower
import pandas as pd

filename = 'breast_cancer.csv'
decision_col = 'diagnosis'
col_to_drop = 'id'

table = pd.read_csv('../data/' + filename)
table = table.drop(columns=[col_to_drop])

col_name_reduct, U_X, U_R = calculate_UX_UR(table, decision_col)
all_upper, all_lower, allaR, allBnd = calculate_upper_lower(U_X, U_R)

print("col_name_reduct: ", col_name_reduct)
print("U_X: ", U_X)
print("U_R: ", U_R)
print("all_lower: ", all_lower)


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
print("all_t: ", all_t)


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
    all_items_in_T = set()

    for t in T:
        all_items_in_T = all_items_in_T.union(all_t[t[0]][t[1]])

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


conclusion = {}
for decision in all_lower:
    conclusion[decision] = []
    B = set(U_X[decision])
    G = copy.deepcopy(B)
    Covering_hollow_T = set()

    while (len(G) != 0):
        condition_T = set()
        all_relevant_T_G = calculate_all_relevant_T_G(G)  # set contains t
        # print("all_relevant_T_G: ", all_relevant_T_G)

        T_belongs_to_B = get_all_item_in_T(condition_T).issubset(B)

        while len(all_relevant_T_G) != 0 and (len(condition_T) == 0 or (not T_belongs_to_B)):  # while T = Ø or [T] Õ/ B
            t = all_relevant_T_G.pop()
            all_relevant_T_G.add(t)
            temp = set()
            temp.add(tuple(t))
            condition_T = condition_T.union(temp)
            G = G.intersection(all_t[t[0]][t[1]])  # G := [t] « G;
            all_relevant_T_G = calculate_all_relevant_T_G(G)
            all_relevant_T_G = all_relevant_T_G.difference(condition_T)  # T(G) := T(G) – T;
            T_belongs_to_B = get_all_item_in_T(condition_T).issubset(B)
        # end inner while

        # for each t in T do
        set_t_to_delete_from_condition_T = []
        for t in condition_T:
            # print("condition_T 1: ", condition_T)
            copy_T = copy.deepcopy(condition_T)
            # print("copy 1: ", copy_T)
            # print("t 1: ", t)
            delete_t_from_T(copy_T, t)
            # print("copy 2: ", copy_T)

            if copy_T != None:
                all_items_in_remove_t = get_all_item_in_T(copy_T)
                # print("copy_T: ", copy_T)
                # print("all_items_in_remove_t: ", all_items_in_remove_t)
                # print("B: ", B)
                if all_items_in_remove_t.issubset(B):
                    print("true")
                    set_t_to_delete_from_condition_T.append(t)

        if len(set_t_to_delete_from_condition_T) != 0:
            print("set_t_to_delete_from_condition_T: ", set_t_to_delete_from_condition_T)
            list_condition_T = list(condition_T)
            print("list_condition_T: before", list_condition_T)
            for t in set_t_to_delete_from_condition_T:
                index = 0
                while index < len(list_condition_T):
                    if list_condition_T[index][0] == t[0] and list_condition_T[index][1] == t[1]:
                        list_condition_T.pop(index)
                    else:
                        index = index + 1
            condition_T = set(list_condition_T)
            print("condition_T: after", condition_T)
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

        print("list_all_item_in_hollowT_minus_T: ", list_all_item_in_hollowT_minus_T)
        print("list_B: ", list_B)
        if list_all_item_in_hollowT_minus_T == list_B:
            print("true 2")
            # print("Covering_hollow_T 1: ", Covering_hollow_T)
            T_to_delete.append(T)
            # print("Covering_hollow_T 2: ", Covering_hollow_T)

    print("T_to_delete: ", T_to_delete)
    print("Covering_hollow_T: before", Covering_hollow_T)
    for T in T_to_delete:
        Covering_hollow_T.remove(T)

    conclusion[decision].append(Covering_hollow_T)

print("Conclusion: ")
for decision in conclusion:
    print("The conditions to achieve the decision of ", decision, " are: ")
    for T in conclusion[decision][0]:
        print(T)
