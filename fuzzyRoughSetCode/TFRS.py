import numpy as np
import pandas as pd
class I_TFRS:## incremental tolerance fuzzy tough set
    def __init__(self):## the use of this tolerance still not decide yet
        pass
    
    ##return the similarity of two object by the attribute id in attr_ids
    #input id1: the id of the first object
    #      id2: the second object
    #      attr_ids: the id of the attributes
    # strtegy: since this is a fuzzy rough set, the similarity aggregation(defined as t_norm in fuzzy set theory) that we could choos from include min, max, and product
    #          the one i pick is the product. So this method just mutiply up the similarity among all the attribute
    # output: tnorm: is the similarity of two object in tese attribute
    def t_relation_per_set(self,id1,id2,attr_ids):
        t_norm = 1
        for attr_id in attr_ids:
            t_norm  = t_norm * self.relation_tensor[attr_id][id1,id2]
        return t_norm
    
    #this method called to generate a relation matrix, using a relation matrix help to dynamically catch the value without calculate it every time
    # input id1: the id of first object
    #       id2: the id of second object
    #       attr_id: the id of the attribute
    # similarity strategy: use guassian similarity, in order to make this similarity roubust, the initial training dataset must as representive as possible
    #                      since the varian only calculate once in the initial dataset
    # way to imporve: one could try to using stream mining method, create a batch for the time window, each window have its own variance
    # output relation: the similarity of these two object in this specific attrbute
    def t_relation_per_attr(self,id1,id2,attr_id):## return the fuzzy relation of x and y
        # relation = np.exp(-((self.X[id1,attr_id] - self.X[id2,attr_id])**2)/(2*self.X_var[attr_id]))
        if self.X[id1, attr_id] == self.X[id2, attr_id]:
            return 1
        else:
            return 0

        # if np.isnan(relation):##just in case variance is 0, then it make sense that they are the same
        #     relation = 1
        # return relation

    #this method is intend to calculate the similarity of nominal attribute and decision
    #assumption are that decision and nominal attribute are all crisp instead of fuzzy
    #now it only used for calculate decision, for nominal attribute calculation, still working
    # return true if they are the same and false if not
    def relation(self,id1,id2,attr_id): ## classical relation
        if attr_id == 'Y':
            return self.Y[id1] == self.Y[id2]
        else:
            return self.X[id1] == self.X[id2]
    
    #this method return the membership function of object id1 to its decision d's lower_approximate
    # input id1: object id
    #       attr_ids: attribute id, in our implementation it will also be the whole attrbute set, 
    #                 just in case for later possible usage, to make it more general
    #strategy: by definition of fuzzy lower approximate: R(low)/d(id1) = inf(max(1-R(id1,id2), m_f/d(id2)),for id2 in self.X)
    #                                                    here R(low)/d(id1) is the memebrship function of id1 belong to decision d's lower approximate
    #                                                    inf <=> minimum   ; max(1-R(x,y), m_f/d(id2)) is a fuzzy indicator; mf/d(id2) is the membership function of id2 to d
    #                                                    please note m_f of id1 to lower approximate of d != m_f/d(id1)
    #                                                    since right now our decision is crisp, m_f/d(x) = 0 if decision of x != d, 1 if == d
    #                                                    from here we could know if decision of id2 is d, it will be one and max(1-R(x,y),1) == 1
    #                                                    so we only need to take care about when decision of id2 is not d, then m_f/d(id2) == 0
    #                                                    then R(low)/d(id1) = inf(max(1-R(id1,id2), 0),for id2 not have same decision as id1)
    #                                                       =>R(low)/d(id1) = inf(max(1-R(id1,id2)),for id2 not have same decision as id1)
    def m_f_to_decision_lower(self,id1,attr_ids):##mf of id1 to [id1]d
        inf_mf = 1
        for id2 in range(self.X.shape[0]):
            if self.relation(id1,id2,'Y') != 1:
                disim = 1 - self.t_relation_per_set(id1,id2,attr_ids)
                if  disim < inf_mf:
                    inf_mf = disim
        return inf_mf
    # this method  generate a matrix indicate those attributes in id1 and id2 with more disimilar degree than the membership function of id1 to its decison's lower approximate 
    # note unlike the regular discernibility matrix, this one is not symmetric, since R(low)/d(id1) != R(low)/d(id2)            
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
        for id1 in range(self.X.shape[0]):
            for id2 in range(self.X.shape[0]):
                dis_set = []
                if self.relation(id1,id2,'Y') != 1:
                    for attr_id in range(self.X.shape[1]):
                        if 1 - self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:# check if more than
                            dis_set.append(attr_id)## if so append
                    dis_mat[id1][id2] = dis_set## set it in the matrix
        return dis_mat

# this method use the discernibility matrix to find : for each attribute, the pair of object is disimilar(regard to first object's membership degree)    
    def discernibility(self):
        dis_dict = {}## create a dictionary
        for attr_id in list(range(self.X.shape[1])):#initial all the attribute with an empty set, the set will stor the pair
            dis_dict[attr_id] = set()
        dis_mat = self.dis_matrix()       
        for row in range(len(dis_mat)):#iterate over the discernibility matrix
            for col in range(len(dis_mat)):
                for attr in dis_mat[row][col]:
                    dis_dict[attr].add((row,col))#add the pair
        return dis_dict  
    
# calculate the relation(similarity) among the object, for each attribute, a catch memory to speed up
# since the dis_matrix will access the similarity frequently
# unlike the discernibility matrix, this relation tensor(3d_matrix) is symmetric in each sub 2_d matrix
    def calculate_relation(self):
        self.relation_tensor = []## a tensor, hold attrbute number of relation matrix
        for attr in range(self.X.shape[1]):
            relation_mat = np.ones((self.X.shape[0],self.X.shape[0]))# for each attribute, generate a matrix, use np.ones instead of np.zero since diognal will always 1
            for id1 in range(self.X.shape[0]-1):# since symmtric, just do upper triangle
                for id2 in range(id1+1,self.X.shape[0]):
                    relation = self.t_relation_per_attr(id1,id2,attr)
                    relation_mat[id1,id2] = relation
                    relation_mat[id2,id1] = relation
            self.relation_tensor.append(relation_mat)

#this method is call when there is new observation coming in
# it will update the relation tensor by adding one more object
# but the similarity regarding to the new object will be calculate else where, actually in the update method
# note the input: by means expand by how much, in out implementation it is always one, but set it to a parameter just in case 

    def relation_expand(self,by = 1):
        temp_tensor = []
        for attr in range(self.X.shape[1]):
            relation_mat = np.ones((self.X.shape[0],self.X.shape[0]))
            for id1 in range(self.X.shape[0]-1-by):
                for id2 in range(id1+1,self.X.shape[0]-by):
                    relation = self.relation_tensor[attr][id1,id2]
                    relation_mat[id1,id2] = relation
                    relation_mat[id2,id1] = relation
            temp_tensor.append(relation_mat)
        self.relation_tensor = temp_tensor

#this method find the reduct of the initial dataset        
    def find_reduct(self):
##################################################################
# step 1: calculate the relation tensor 
        self.calculate_relation()
##################################################################
# step 2: calculate the membership function of each object to its decision class's lower approximation
        self.m_f = []
        for id1 in range(self.X.shape[0]):
            self.m_f.append(self.m_f_to_decision_lower(id1,list(range(self.X.shape[1]))))
##################################################################
# step 3: calculate the discernibility set [(x1,x2)] for each attribute and the combination of each attribute
#         the discernibility of the combination set of all the attribute is the union of discernibility of each attributes
        self.dis_dict = self.discernibility()
        self.dis_all = set()## since it is a set, add duplicate to it will get ignore
        for key in self.dis_dict:
            for pair in self.dis_dict[key]:
                self.dis_all.add(pair)
#################################################################
# step 4: calculate the core of the initial dataset
# we do this by calculate the discernibility of A - a for all a belong A (A denote the combination of all attribute and a is the attributes in A)
# then if there is an 'a' such that dis(A - a) != dis(A), a should be the in the core
        core = []
        for attr_id in list(range(self.X.shape[1])):
            dis_exclude_a = set() #calculate the discernibility of A - a for all a belong A
            for key in self.dis_dict:
                if key == attr_id:
                    continue
                for pair in self.dis_dict[key]: # pair is (id1,id2) tuple, use tuple as it could be hashed
                    dis_exclude_a.add(pair)
            if dis_exclude_a != self.dis_all:#dis(A - a) != dis(A)
                core.append(attr_id) #a should be the in the core   
#################################################################
# step 5, calculate the reduct
# initialize the reduct with core, calculate the dis(reduct) and for the discenibility set of each attribute, exclude those in the dis(reduct) 
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)
        red = core#initialize the reduct with core
        dis_red = set()

        for attr_id in red: #calculate the dis(reduct)
            value = self.dis_dict[attr_id]
            for pair in value:
                dis_red.add(pair)
        #for the discenibility set of each attribute, exclude those in the dis(reduct)       
        for attr_id in list(range(self.X.shape[1])):
            if not attr_id in red:
                self.dis_dict[attr_id] = self.dis_dict[attr_id].difference(dis_red)
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)                
        while self.dis_all != dis_red:
            max_size = -1
            candidate = -1
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    if len(self.dis_dict[attr_id]) > max_size:
                        max_size = len(self.dis_dict[attr_id])
                        candidate = attr_id
            red.append(candidate)
            dis_red = dis_red.union(self.dis_dict[candidate])
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    self.dis_dict[attr_id] = self.dis_dict[attr_id].difference(dis_red)
        return red
            
    def fit(self, decision_table, decision_col_name):
        self.Y = decision_table[decision_col_name].values
        self.X = decision_table.drop([decision_col_name], axis = 1).values
        self.reduct_attr = self.find_reduct()
# this return the membership function of newX belong to decision's lower approximate
# this method was used for prediction
# note that once we get the reduct, we ignore other attribute for prediction we will only compare the similarity in the reduct(see next method)
    def mf(self,newX,decision):
        inf_mf = 1
        for id2 in range(self.X.shape[0]):
            if self.Y[id2] != decision:
                disim = 1 - self.similarity(newX,self.X[id2])
                if  disim < inf_mf:
                    inf_mf = disim
        return inf_mf
# return the similarity of x and y regarding the reduct attrbuite    
    def similarity(self,x,y):
        tnorm = 1
        for attr in self.reduct_attr:
            relation = np.exp(-(y[attr] - x[attr])**2/(2*self.X_var[attr]))
            if np.isnan(relation):##just in case variance is 0, then it make sense that they are the same 
                relation = 1
            tnorm = tnorm * relation
            
        return tnorm
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
# step 1: initialize the new reduct by the current reduct
        new_reduct = self.reduct_attr.copy()
        dis_red= set()
        #then calculate the dis(new_reduct)
        for attr_id in new_reduct:
            pairs = self.dis_dict[attr_id]
            for pair in pairs:
                dis_red.add(pair)
#########################################################
# step 2 if dis(new_red) == dis_all go step 4
        if dis_red != self.dis_all:
#########################################################
# step 3 like what we do before, find the attribute with highest discernibility add it to reduct, until we get dis(new_red) == dis_all
            while dis_red != self.dis_all:
                dis_union_all = {}
                for attr_id in range(self.X.shape[1]):
                    if not attr_id in new_reduct:
                        dis_union_a = dis_red.union(self.dis_dict[attr_id])
                        dis_union_all[attr_id] = dis_union_a
                max_key = -1
                max_cover = 0
                for key in dis_union_all:
                    cover = len(dis_union_all[key])
                    if cover > max_cover:
                        max_cover = cover
                        max_key = key
                new_reduct.append(max_key)
                dis_red = dis_union_all[max_key]
########################################################
# step 4 new_red could be a reduct or superset of reduct, need to check whether there is redundant attribute               
        while dis_red == self.dis_all:
            dis_exclude_all = {}
            #calculate the discernibility after remove one attribute from the new_reduct
            for a in new_reduct:
                dis_exclude_a = set()
                for attr in new_reduct:
                    if attr == a:
                        continue
                    else:
                        value = self.dis_dict[attr]
                        for pair in value:
                            dis_exclude_a.add(pair)
                dis_exclude_all[a] = dis_exclude_a
            found = False
            #check whether there is attribute get deleted, and the rest attributes is still a reduct
            for attr_key in dis_exclude_all:
                if dis_exclude_all[attr_key] == self.dis_all:
                    new_reduct.remove(attr_key)
                    found = True
                    break
            if not found:
                break
        self.reduct_attr = new_reduct
                    
#this method will update decision table after there is new instance coming in 
# step0 add the new instance to X and Y
# step1 expand the relation tensor by one col and row
#       then calculate similarity in the expanded row and col, this is the similarity of the new instance to the one exist in the X(universe) 
# step2 calculate the membership function of the new instance to its decision class's lower approximate
#       update the membership function of each instance to their decision class's lower approximate   
#       strategy for last row: we know m_f/d(x) <= 1 - R(x,y) for those y not agree with x in decision, here x denote the exist instance(not the new one)
#                              then if decision of x == decision of newX, then m_f/d(x) is unchanged
#                              if not agree on decision then m_f/d(x) = min(m_f/d(x), 1 - R(x,newX))
# step3 update the discernibility set for all attribute this step only update those including newX
# for each id1 in the old universe(before newX comein)
#        for attribute a:
#            if 1 - R/a(id1,newX) >= m_f/d(id1):  add (id1, newX) to dis(a), add(id1,newX) to dis_all 
#            if 1 - R/a(id1,newX) >= m_f/d(newX):  add (newX,id1) to dis(a), add(newX,id1) to dis_all     
#            just keep in mind that as before the dis_matrix is not symmetric, same here, we need to keep both
# step4 update the discernibility set for all attribute this step update those not include newX, as newX come in, things change
#       since we update some membership degree when new instance comes in, there is a new threshold
#       and also remember we drop some pair when we calculate the reduct for initial dataset?
# for each instance pair (id1,id2):
#      for each attribute a:
#          if (id1,id2) not exist in dis(a) and d(id1) != d(id2) amd d(id1) != d(newX)
#                here d(id1) != d(newX) means m_f/d(id1) might change in the last step, we need to update
#                     d(id1) != d(id2) means they will be discerned by some attribute
#                and add the pair like what we do before for the initial data                
                   
    def update(self,newX,newY):#this x should contain the decision as well
        # step0 add the new instance to X and Y
        self.Y = np.append(self.Y,newY)
        self.X = np.row_stack((self.X,newX))
        # step1 expand the relation tensor by one col and row
        self.relation_expand()
        # calculate similarity in the expanded row and col
        for attr_id in range(self.X.shape[1]):
            for id1 in range(self.X.shape[0]-1):
                sim = self.t_relation_per_attr(id1,-1,attr_id)
                self.relation_tensor[attr_id][id1,-1] = sim
                self.relation_tensor[attr_id][-1,id1] = sim
        #calculate the membership function of the new instance to its decision class's lower approximate
        self.m_f.append(self.m_f_to_decision_lower(-1,list(range(self.X.shape[1]))))#note index -1 means the last element
#       update the membership function of each instance to their decision class's lower approximate   
#       strategy for last row: we know m_f/d(x) <= 1 - R(x,y) for those y not agree with x in decision, here x denote the exist instance(not the new one)
#                              then if decision of x == decision of newX, then m_f/d(x) is unchanged
#                              if not agree on decision then m_f/d(x) = min(m_f/d(x), 1 - R(x,newX))        
        for id1 in range(self.X.shape[0]-1):
            if self.Y[id1] == newY:
                pass
            else:
                dism = 1 - self.t_relation_per_set(id1,-1,list(range(self.X.shape[1])))
                if self.m_f[id1] <= dism:
                    pass
                else:
                    self.m_f[id1] = dism
# step3 update the discernibility set for all attribute this step only update those including newX
# for each id1 in the old universe(before newX comein)
#        for attribute a:
#            if 1 - R/a(id1,newX) >= m_f/d(id1):  add (id1, newX) to dis(a), add(id1,newX) to dis_all 
#            if 1 - R/a(id1,newX) >= m_f/d(newX):  add (newX,id1) to dis(a), add(newX,id1) to dis_all     
#            just keep in mind that as before the dis_matrix is not symmetric, same here, we need to keep both                    
        for id1 in range(self.X.shape[0]-1):
            for attr_id in self.dis_dict:
                if 1- self.relation_tensor[attr_id][id1,-1] >= self.m_f[id1]:
                    self.dis_dict[attr_id].add((id1,self.X.shape[0]-1))
                    self.dis_all.add((id1,self.X.shape[0]-1))
                if 1- self.relation_tensor[attr_id][id1,-1] >= self.m_f[self.X.shape[0]-1]:
                    self.dis_dict[attr_id].add((self.X.shape[0]-1,id1))
                    self.dis_all.add((self.X.shape[0]-1,id1))
# step4 update the discernibility set for all attribute this step update those not include newX, as newX come in, things change
#       since we update some membership degree when new instance comes in, there is a new threshold
#       and also remember we drop some pair when we calculate the reduct for initial dataset?
# for each instance pair (id1,id2):
#      for each attribute a:
#          if (id1,id2) not exist in dis(a) and d(id1) != d(id2) amd d(id1) != d(newX)
#                here d(id1) != d(newX) means m_f/d(id1) might change in the last step, we need to update
#                     d(id1) != d(id2) means they will be discerned by some attribute
#                and add the pair like what we do before for the initial data   
        for attr in self.dis_dict:
            for id1 in range(self.X.shape[0]):
                for id2 in range(id1,self.X.shape[0]):
                    if not (id1,id2) in self.dis_dict[attr]:
                        if self.Y[id1] != self.Y[id2] and self.Y[id1] != self.Y[-1]:
                            if 1 - self.relation_tensor[attr][id1,id2] >= self.m_f[id1]:
                                 self.dis_dict[attr_id].add((id1,id2))
                                 self.dis_all.add((id1,id2))                        
        self.update_reduct()                
                
## predict the decion of the newX
## simply calculate the membership degree of newX to each dicision class's lower approximate
## and the one with higher degree will get predicted                   
    def predict(self,newX):
        max_mf = 0
        best_decision = None
        for decision in np.unique(self.Y):
            mf_d = self.mf(newX,decision)
            if mf_d > max_mf:
                max_mf = mf_d
                best_decision = decision
        return best_decision
