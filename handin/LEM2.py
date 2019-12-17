import numpy as np
#lem2 algorithm
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
        for attr_id in reduct:# for each attribute in the reduct
            all_t[attr_id] = {}# a dictionary that will hold different value of current attribite as key and the index with this value as value
            unique_values = np.unique(X[:,attr_id])#all possible value in attribute_id th dttribute
            for value in unique_values:
                all_t[attr_id][value] = set()# create an empty set which will hold index later
        return all_t

    def populate_all_t(self,X,Y,reduct):
        for id1 in range(X.shape[0]):
            for attr_id in reduct:
                self.all_t[attr_id][X[id1,attr_id]].add(id1)# add the current id to the correspond set
    #this routine calculate all the possible candidate attribute:value pair in the current set, which will be in the rule later
    # input G: the object set need to make some rules based on
    def calculate_all_relevant_T_G(self,G):
        all_relevant_T_G = set()   # a set that hold all the possible candidate attribute: value pair 
        for condition in self.all_t:# for each attribute in reduct
            for value in self.all_t[condition]: # for each possible value in current attribute
                intersection = self.all_t[condition][value].intersection(G) #the intersection of [object set that meet this condition] with G
                if len(intersection) != 0:# if not empty, that means this value attribute value pair could be a candidate
                    t = [condition, value]# make a pair
                    all_relevant_T_G.add(tuple(t))# add it to candidate set
    
        return all_relevant_T_G

    #T: a rule
    def get_all_item_in_T(self,T):
        all_set_t = []
        for t in T: # for each condition in the rule
            all_set_t.append(self.all_t[t[0]][t[1]])#append all the id of those objects that meet this condition   
        all_items_in_T = set()  #empty set  
        if len(all_set_t) != 0:# if there is at least one objects that could meet one of condition in T
            all_items_in_T = all_set_t[0]# initialize it with objects meet the first condition
            for t in all_set_t:
                all_items_in_T = all_items_in_T.intersection(t)# intersection to the object that meet the next rule
    
        return all_items_in_T# the result will be a set of object that meet the rule T

    #hollow_T: all the rule mined so far
    def get_all_item_in_hollow_T(self,hollow_T):
        all_items_in_hollow_T = set()    
        for T in hollow_T:#get the union of all the object that meet rule in hollow_t
            all_items_in_hollow_T = all_items_in_hollow_T.union(self.get_all_item_in_T(T))    
        return all_items_in_hollow_T

    #delete the condition from the rule
    def delete_t_from_T(self,T, t):
        for pair in T:
            if pair[0] == t[0] and pair[1] == t[1]:
                T.remove(t)
                break
    #delete the rule from the rules set
    def delete_T_from_hollow_T(self,hollow_T, T_to_delete):
        copy_hollow_T = hollow_T.copy()     
        copy_hollow_T.remove(T_to_delete)
        return copy_hollow_T
    #select the best attribute
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
    #lem2 algorithm
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
                    temp = set()
                    temp.add(tuple(t))
                    condition_T = condition_T.union(temp)
                    G = G.intersection(self.all_t[t[0]][t[1]])  # G := [t] « G;
                    all_relevant_T_G = self.calculate_all_relevant_T_G(G)
                    all_relevant_T_G = all_relevant_T_G.difference(condition_T)  # T(G) := T(G) – T;
                    T_belongs_to_B = self.get_all_item_in_T(condition_T).issubset(B)
                copy_ = condition_T.copy()
                for t in copy_:
                    copy_T = condition_T.copy()
                    self.delete_t_from_T(copy_T, t)   
                    if copy_T != None and len(copy_T) != 0:
                        all_items_in_remove_t = self.get_all_item_in_T(copy_T)
                        if all_items_in_remove_t.issubset(B):                       
                            condition_T = copy_T
                temp = set()
                temp.add(tuple(condition_T))
                Covering_hollow_T = Covering_hollow_T.union(temp)
                G = B.difference(self.get_all_item_in_hollow_T(Covering_hollow_T))
                
                count = count + 1

            copy_ = Covering_hollow_T.copy()
            for T in copy_:
                hollowT_minus_T = self.delete_T_from_hollow_T(Covering_hollow_T, T)    
                all_item_in_hollowT_minus_T = self.get_all_item_in_hollow_T(hollowT_minus_T)
                list_all_item_in_hollowT_minus_T = list(all_item_in_hollowT_minus_T)
                list_all_item_in_hollowT_minus_T.sort()
                list_B = list(B)
                list_B.sort()
                if list_all_item_in_hollowT_minus_T == list_B:

                     Covering_hollow_T.remove(T)
            conclusion[decision] = Covering_hollow_T
        return conclusion





