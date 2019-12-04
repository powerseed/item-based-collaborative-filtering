import numpy as np
class LEM2:
    def induce_rule(self,X,Y,reduct,lower_record,upper_record):
        self.X_shape = X.shape[0]
        self.all_t = self.get_all_t(X,Y,reduct)
        self.populate_all_t(X,Y,reduct)
        possible_rule = self.lem2(upper_record)
        fix_rule = self.lem2(lower_record)
        return fix_rule,possible_rule
        
    def get_all_t(self,X,Y,reduct):
        all_t = {}
        for attr_id in reduct:
            all_t[attr_id] = {}
            unique_values = np.unique(X[:,attr_id])
            for value in unique_values:
                all_t[attr_id][value] = set()
        return all_t

    def populate_all_t(self,X,Y,reduct):
        for id1 in range(X.shape[0]):
            for attr_id in reduct:
                self.all_t[attr_id][X[id1,attr_id]].add(id1)

    def calculate_all_relevant_T_G(self,G):
        all_relevant_T_G = set()    
        for condition in self.all_t:
            for value in self.all_t[condition]:
                intersection = self.all_t[condition][value].intersection(G)
                if len(intersection) != 0:
                    t = [condition, value]
                    all_relevant_T_G.add(tuple(t))
    
        return all_relevant_T_G


    def get_all_item_in_T(self,T):
        all_set_t = []
        for t in T:
            all_set_t.append(self.all_t[t[0]][t[1]])
    
        all_items_in_T = set()
    
        if len(all_set_t) != 0:
            all_items_in_T = all_set_t[0]
            for t in all_set_t:
                all_items_in_T = all_items_in_T.intersection(t)
    
        return all_items_in_T


    def get_all_item_in_hollow_T(self,hollow_T):
        all_items_in_hollow_T = set()    
        for T in hollow_T:
            all_items_in_hollow_T = all_items_in_hollow_T.union(self.get_all_item_in_T(T))
    
        return all_items_in_hollow_T


    def delete_t_from_T(self,T, t):
        for pair in T:
            if pair[0] == t[0] and pair[1] == t[1]:
                T.remove(t)
                break

    def delete_T_from_hollow_T(self,hollow_T, T_to_delete):
        copy_hollow_T = hollow_T.copy()     
        copy_hollow_T.remove(T_to_delete)
        return copy_hollow_T

    def select_a_t(self,all_relevant_T_G, G):
        list_all_relevant_T_G = list(all_relevant_T_G)
        list_cardinality = []
    
        index = 0
        while index < len(list_all_relevant_T_G):
            t = list_all_relevant_T_G[index]
            set_t = self.all_t[t[0]][t[1]]
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
                cardinality_t = len(self.all_t[t[0]][t[1]])
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

    def lem2(self,lower_or_upper_set):
        conclusion = {}
        for decision in lower_or_upper_set:
            conclusion[decision] = []
            B = lower_or_upper_set[decision]
            G = B.copy()
            Covering_hollow_T = set()
    
            count = 0
            while (len(G) != 0):
                condition_T = set()
                all_relevant_T_G = self.calculate_all_relevant_T_G(G)  # set contains t
                # print("all_relevant_T_G: ", all_relevant_T_G)
    
                T_belongs_to_B = self.get_all_item_in_T(condition_T).issubset(B)
    
                while len(all_relevant_T_G) != 0 and (len(condition_T) == 0 or (not T_belongs_to_B)):  # while T = Ø or [T] Õ/ B
                    t = self.select_a_t(all_relevant_T_G, G)
                    all_relevant_T_G.remove(t)
                    # print("condition_T 1: ", condition_T)
                    # print("t 1: ", t)
                    temp = set()
                    temp.add(tuple(t))
                    condition_T = condition_T.union(temp)
                    G = G.intersection(self.all_t[t[0]][t[1]])  # G := [t] « G;
                    all_relevant_T_G = self.calculate_all_relevant_T_G(G)
                    all_relevant_T_G = all_relevant_T_G.difference(condition_T)  # T(G) := T(G) – T;
                    T_belongs_to_B = self.get_all_item_in_T(condition_T).issubset(B)
                # end inner while
    
                # if len(condition_T) == 1:
                #     print("condition_T: ", condition_T, " count: ", count)
                #     break
                # for each t in T do
                #set_t_to_delete_from_condition_T = []
                copy_ = condition_T.copy()
                for t in copy_:
                    # print("condition_T 1: ", condition_T)
                    copy_T = condition_T.copy()
                    # print("copy 1: ", copy_T)
                    # print("t 1: ", t)
                    self.delete_t_from_T(copy_T, t)
                    # print("copy 2: ", copy_T)    
                    if copy_T != None and len(copy_T) != 0:
                        all_items_in_remove_t = self.get_all_item_in_T(copy_T)
                        if all_items_in_remove_t.issubset(B):                       
                            condition_T = copy_T
#                if len(set_t_to_delete_from_condition_T) != 0:
#                    # print("set_t_to_delete_from_condition_T: ", set_t_to_delete_from_condition_T)
#                    list_condition_T = list(condition_T)
#                    # print("list_condition_T: before", list_condition_T)
#                    for t in set_t_to_delete_from_condition_T:
#                        index = 0
#                        while index < len(list_condition_T):
#                            if list_condition_T[index][0] == t[0] and list_condition_T[index][1] == t[1]:
#                                list_condition_T.pop(index)
#                            else:
#                                index = index + 1
#                    condition_T = set(list_condition_T)
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
                G = B.difference(self.get_all_item_in_hollow_T(Covering_hollow_T))
                # print("G1: after", G)
                count = count + 1
            # outer while
            copy_ = Covering_hollow_T.copy()
            for T in copy_:
                # print("Covering_hollow_T: ", Covering_hollow_T)
                # print("T: ", T)
                hollowT_minus_T = self.delete_T_from_hollow_T(Covering_hollow_T, T)
                # print("hollowT_minus_T: ", hollowT_minus_T)
    
                all_item_in_hollowT_minus_T = self.get_all_item_in_hollow_T(hollowT_minus_T)
                list_all_item_in_hollowT_minus_T = list(all_item_in_hollowT_minus_T)
                list_all_item_in_hollowT_minus_T.sort()
                list_B = list(B)
                list_B.sort()
    
                # print("list_all_item_in_hollowT_minus_T: ", list_all_item_in_hollowT_minus_T)
                # print("list_B: ", list_B)
                if list_all_item_in_hollowT_minus_T == list_B:
                    # print("Covering_hollow_T 1: ", Covering_hollow_T)
                     Covering_hollow_T.remove(T)
            conclusion[decision] = Covering_hollow_T
        return conclusion





