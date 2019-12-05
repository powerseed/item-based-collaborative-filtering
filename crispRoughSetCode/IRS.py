import numpy as np
import LEM2 as lem2
class I_RS:## incremental tolerance fuzzy tough set
    def __init__(self):## the use of this tolerance still not decide yet
        pass
    
    def dis_matrix(self):
#############################################################
#step 1: create an empty matrix
        dis_mat = []
        for i in range(self.X.shape[0]):
            row = []
            for j in range(self.X.shape[0]):
                row.append([])
            dis_mat.append(row)
##############################################################
#step 2: for each entry in the matrix, add those attrbute are disimilar more than m_f(id1)
        for id1 in range(self.X.shape[0]-1):
            for id2 in range(id1+1,self.X.shape[0]):
                dis_set = []
                if self.Y[id1] != self.Y[id2]:
                    for attr_id in range(self.X.shape[1]):
                        if self.X[id1,attr_id] != self.X[id2,attr_id]:# left side euqal to 0 or 1 and right side equal to 0 or 1, since it is a crisp rough set 
                            dis_set.append(attr_id)## if so append
                    dis_mat[id1][id2] = dis_set## set it in the matrix
        return dis_mat

# this method use the discernibility matrix to find : for each attribute, the pair of object is disimilar(regard to first object's membership degree)    
    def discernibility(self):
        dis_dict = {}## create a dictionary
        for attr_id in list(range(self.X.shape[1])):#initial all the attribute with an empty set, the set will stor the pair
            dis_dict[attr_id] = set()
        dis_mat = self.dis_matrix()       
        for row in range(1,len(dis_mat)-1):#iterate over the discernibility matrix
            for col in range(row,len(dis_mat)):
                for attr in dis_mat[row][col]:
                    dis_dict[attr].add((row,col))#add the pair
        return dis_dict  
    
#this method find the reduct of the initial dataset        
    def find_reduct(self):
##################################################################
# step 1: calculate the discernibility set [(x1,x2)] for each attribute and the combination of each attribute
#         the discernibility of the combination set of all the attribute is the union of discernibility of each attributes
        self.dis_dict = self.discernibility()
        self.dis_all = set()## since it is a set, add duplicate to it will get ignore
        for key in self.dis_dict:
            for pair in self.dis_dict[key]:
                self.dis_all.add(pair)
#################################################################
# step 2: calculate the core of the initial dataset
# we do this by calculate the discernibility of A - a for all a belong A (A denote the combination of all attribute and a is the attributes in A)
# then if there is an 'a' such that dis(A - a) != dis(A), a should be the in the core
        core = []
        for attr_id in list(range(self.X.shape[1])):
            dis_exclude_a = set() #calculate the discernibility of A - a for all a belong A
            for key in self.dis_dict:
                if key == attr_id:
                    continue
                else:
                    dis_exclude_a = dis_exclude_a.union(self.dis_dict[key])
            if dis_exclude_a != self.dis_all:#dis(A - a) != dis(A)
                core.append(attr_id) #a should be the in the core   
#################################################################
# step 3, calculate the reduct
# initialize the reduct with core, calculate the dis(reduct) and for the discenibility set of each attribute, exclude those in the dis(reduct) 
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)
        red = core#initialize the reduct with core
        self.dis_red = set()

        for attr_id in red: #calculate the dis(reduct)
            self.dis_red = self.dis_red.union(self.dis_dict[attr_id])           
        #for the discenibility set of each attribute, exclude those in the dis(reduct)  
        copy_dis_dict = self.dis_dict.copy()
        for attr_id in list(range(self.X.shape[1])):
            if not attr_id in red:
                copy_dis_dict[attr_id] = copy_dis_dict[attr_id].difference(self.dis_red)
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)                
        while self.dis_all != self.dis_red:
            max_size = -1
            candidate = -1
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    if len(copy_dis_dict[attr_id]) > max_size:
                        max_size = len(copy_dis_dict[attr_id])
                        candidate = attr_id
            red.append(candidate)
            self.dis_red = self.dis_red.union(copy_dis_dict[candidate])
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    copy_dis_dict[attr_id] = copy_dis_dict[attr_id].difference(self.dis_red)
        
        return red
    def size(self):
        return self.X.shape[0]
    def fit(self, X, Y):
        self.Y = Y
        self.X = X
        self.rule_miner = lem2.LEM2()
        self.reduct_attr = self.find_reduct()
        U_X,U_R = self.calculate_UX_UR()
        self.calculate_positive_boundary(U_X,U_R)        
        fix_rule,possible_rule = self.rule_miner.induce_rule(self.X,self.Y,self.reduct_attr,self.all_lower_records,self.all_bound_records)
        self.fix_rule = self.find_rule_coverage(fix_rule)
        self.possible_rule = self.find_rule_coverage(possible_rule)
# this method update the reduct when there is new object getting in
# note the new object already added into the self.X, access it by self.X[-1]
# step 1 this method initialize the new reduct by the current reduct, then calculate the dis(new_reduct)
# note that once object getting in, discernibility of each attribute will changed, the method that change them is in next method
# now check if dis_red = dis_all, (note the dis_all is also get changed already in the next method)
# step 2 if dis(new_red) == dis_all go step 4
# step 3 if dis(new_red) != dis_all
#        like what we do before, find the attribute with highest discernibility add it to reduct, until we get dis(new_red) == dis_all
# step 4 then new_red could be a reduct or superset of reduct, need to check whether there is redundant attribute
#        we will check whether there is attribute get deleted, and the rest attributes is still a reduct, if so, delete this attribute
#        keep doing this until there is no such attribute, we get the new reduct
    def update_reduct(self):
#########################################################
# step 1 if dis(new_red) == dis_all go step 4
        if self.dis_red != self.dis_all:
#########################################################
# step 3 like what we do before, find the attribute with highest discernibility add it to reduct, until we get dis(new_red) == dis_all
            while self.dis_red != self.dis_all:
                dis_union_all = {}
                for attr_id in range(self.X.shape[1]):
                    if not attr_id in self.reduct_attr:
                        dis_union_a = self.dis_red.union(self.dis_dict[attr_id])
                        dis_union_all[attr_id] = dis_union_a
                max_key = -1
                max_cover = 0
                for key in dis_union_all:
                    cover = len(dis_union_all[key])
                    if cover > max_cover:
                        max_cover = cover
                        max_key = key
                self.reduct_attr.append(max_key)
                self.dis_red = dis_union_all[max_key]
########################################################
# step 4 new_red could be a reduct or superset of reduct, need to check whether there is redundant attribute               
        while self.dis_red == self.dis_all:
            dis_exclude_all = {}
            #calculate the discernibility after remove one attribute from the new_reduct
            for a in self.reduct_attr:
                dis_exclude_a = set()
                for attr in self.reduct_attr:
                    if attr == a:
                        continue
                    else:
                        dis_exclude_a = dis_exclude_a.union(self.dis_dict[attr])
                dis_exclude_all[a] = dis_exclude_a
            #check whether there is attribute get deleted, and the rest attributes is still a reduct
            candidate_remove = []
            for attr_key in dis_exclude_all:
                if dis_exclude_all[attr_key] == self.dis_all:
                    candidate_remove.append(attr_key)
            if len(candidate_remove) == 0:
                break
            elif len(candidate_remove) > 1:                
                min_key = -1
                min_cover = np.inf
                for i in range(len(candidate_remove)):
                    if len(self.dis_dict[candidate_remove[i]])< min_cover:
                        min_key = candidate_remove[i]
                        min_cover = len(self.dis_dict[candidate_remove[i]])
                self.reduct_attr.remove(min_key)
            else:
                self.reduct_attr.remove(candidate_remove[0])
                                     
    def update(self,newX,newY):#this x should contain the decision as well
        # step0 add the new instance to X and Y
        self.Y = np.append(self.Y,newY)
        self.X = np.row_stack((self.X,newX))                  
        for id1 in range(self.X.shape[0]-1):
            for attr_id in self.dis_dict:
                if self.Y[id1] != self.Y[-1]:
                    if self.X[-1,attr_id] != self.X[id1,attr_id]:
                        self.dis_dict[attr_id].add((id1,self.X.shape[0]-1))
                        self.dis_dict[attr_id].add((self.X.shape[0]-1,id1))
                        self.dis_all.add((id1,self.X.shape[0]-1))
                        self.dis_all.add((self.X.shape[0]-1,id1))
                        if attr_id in self.reduct_attr:
                            self.dis_red.add((id1,self.X.shape[0]-1))
        self.update_reduct()
        U_X,U_R = self.calculate_UX_UR()
        self.calculate_positive_boundary(U_X,U_R)        
        fix_rule,possible_rule = self.rule_miner.induce_rule(self.X,self.Y,self.reduct_attr,self.all_lower_records,self.all_bound_records)
        self.fix_rule = self.find_rule_coverage(fix_rule)
                
    def update_group(self,newX,newY):
        self.Y = np.append(self.Y,newY)
        self.X = np.row_stack((self.X,newX))
        for id1 in range(self.X.shape[0] - newX.shape[0]):
            for id2 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
                for attr_id in self.dis_dict:
                    if self.Y[id1] != self.Y[id2]:
                        if self.X[id1,attr_id] != self.X[id2,attr_id]:
                            self.dis_dict[attr_id].add((id1,id2))
                            self.dis_all.add((id1,id2))
                            if attr_id in self.reduct_attr:
                                self.dis_red.add((id1,id2))                               
        for attr_id in self.dis_dict:
            for id1 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]-1):
                for id2 in range(id1, self.X.shape[0]):
                    if self.Y[id1] != self.Y[id2]:
                        if self.X[id1,attr_id] != self.X[id2,attr_id]:
                            self.dis_dict[attr_id].add((id1,id2))
                            self.dis_all.add((id1,id2))
                            if attr_id in self.reduct_attr:
                                self.dis_red.add((id1,id2))
        self.update_reduct()
        U_X,U_R = self.calculate_UX_UR()
        self.calculate_positive_boundary(U_X,U_R)        
        fix_rule,possible_rule = self.rule_miner.induce_rule(self.X,self.Y,self.reduct_attr,self.all_lower_records,self.all_bound_records)
        self.fix_rule = self.find_rule_coverage(fix_rule)
        self.possible_rule = self.find_rule_coverage(possible_rule)
        #self.possible_rule = self.find_rule_coverage(possible_rule)
################################################################################################################################
#calculate upper and lower
    def calculate_UX_UR(self):
        values_decision = np.unique(self.Y)
        U_X = {}
        for value in values_decision:
            U_X[value] = np.where(self.Y == value)[0]
        U_R = {}
        for id1 in range(self.X.shape[0]):
            values_conditions = ''
            index_col_name_reduct = 0
            for attr_id in self.reduct_attr:
                values_conditions = values_conditions + str(self.X[id1,attr_id])

                if index_col_name_reduct != len(self.reduct_attr) - 1:
                    values_conditions = values_conditions + ", "

            index_col_name_reduct = index_col_name_reduct + 1
            if not values_conditions in U_R:
               U_R[values_conditions] = []

            U_R[values_conditions].append(id1)   
        return U_X, U_R     
    def calculate_positive_boundary(self,U_X,U_R):
        self.all_bound = {}
        self.all_bound_records = {}
    
        self.all_lower_approximate = {}
        self.all_lower_records = {}
    
        allaR = []
        allBnd = []
    
        for decision in U_X:
            lowerap = []
            upperap = []
    
            lowerap_records = set()
            upperap_records = set()
    
            U_Xcon = set(U_X[decision])
            for condition in U_R:
                U_Rcon = set(U_R[condition])
                # if decision is a subset of result
                if (U_Rcon.issubset(U_Xcon)):
                    lowerap.append(condition)
                    lowerap_records = lowerap_records.union(U_Rcon)
                    continue
                # if decision contains element of result
                if (U_Rcon.isdisjoint(U_Xcon) == False):
                    upperap.append(condition)
                    upperap_records = upperap_records.union(U_Rcon)
    
            #self.all_lower_approximate[decision] = lowerap
            self.all_lower_records[decision] = lowerap_records
    
            #self.all_bound[decision] = upperap
            self.all_bound_records[decision] = upperap_records
    
            # allaR.add(len(lowerap)/len(upperap)) #粗糙度
        # allBnd.append(lowerap - upperap)#bond    
    
    def find_rule_coverage(self,all_rule):
        rule_coverage = []
        for decision in all_rule:
            rules = all_rule[decision]
            for rule in rules:
                cover = set()
                for i in range(self.X.shape[0]):
                    cover.add(i)
                for condition in rule:
                    cover = cover.intersection(set(np.where(self.X[:,condition[0]] == condition[1])[0]))
                match_decision_count = 0
                for id1 in cover:
                    if self.Y[id1] == decision:
                        match_decision_count = match_decision_count+1
                rule_coverage.append((rule,decision,len(cover),match_decision_count/len(cover)))
        return rule_coverage
    
    def predict(self,newX):
        match_decision = {}
        for rule in self.fix_rule:
           condition = rule[0]
           match = True
           for pair in condition:
               if newX[pair[0]] != pair[1]:
                   match = False
                   break
           if match:
               if rule[1] in match_decision:
                   match_decision[rule[1]][0] = match_decision[rule[1]][0]+rule[2]
                   match_decision[rule[1]][1] = match_decision[rule[1]][1]+rule[3]
               else:
                   match_decision[rule[1]] = [rule[2],rule[3]]
        if len(match_decision) == 1:
           for decision in match_decision:
               return decision,match_decision[decision][0]
        elif len(match_decision) > 1:
           max_cover = 0
           final_decision = None
           for decision in match_decision:
               if match_decision[decision][0] > max_cover:
                   max_cover = match_decision[decision][0]
                   final_decision = decision
           return final_decision,max_cover
        elif len(match_decision) == 0:
           #print('Unknow') 
           return self.predict_by_possible_rule(newX)
    def predict_by_possible_rule(self,newX):
        match_decision = {}
        for rule in self.possible_rule:
           condition = rule[0]
           match = True
           for pair in condition:
               if newX[pair[0]] != pair[1]:
                   match = False
                   break
           if match:
               if rule[1] in match_decision:
                   match_decision[rule[1]][0] = match_decision[rule[1]][0]+rule[2]
                   match_decision[rule[1]][1] = match_decision[rule[1]][1]+rule[3]
               else:
                   match_decision[rule[1]] = [rule[2],rule[3]]
        if len(match_decision) == 1:
           for decision in match_decision:
               return decision,match_decision[decision][0]*match_decision[decision][1]
        elif len(match_decision) > 1:
           max_cover = 0
           final_decision = None
           for decision in match_decision:
               if match_decision[decision][0] > max_cover:
                   max_cover = match_decision[decision][0]*match_decision[decision][1]
                   final_decision = decision
           return final_decision,max_cover
        elif len(match_decision) == 0:
           #print('Unknow') 
           return 'Unknow',0
       