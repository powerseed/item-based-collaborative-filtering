import numpy as np
class I_RS:## incremental tolerance fuzzy tough set
    def __init__(self):## the use of this tolerance still not decide yet
        pass
    
    ##return the similarity of two object by the attribute id in attr_ids
    #input id1: the id of the first object
    #      id2: the second object
    #      attr_ids: the id of the attributes
    # strtegy: since this is a fuzzy rough set, the similarity aggregation(defined as t_norm in fuzzy set theory) that we could choos from include min, max, and product
    #          the one we pick is the product. So this method just mutiply up the similarity among all the attribute
    # output: tnorm: is the similarity of two object in tese attribute
    def relation_per_set(self,id1,id2,attr_ids):
        sim = 1
        for attr_id in attr_ids:
            sim  = sim * self.relation_tensor[attr_id][id1,id2]
        return sim
    
    
    def relation_per_attr(self,id1,id2,attr_id):## return the fuzzy relation of x and y
        if self.X[id1] == self.X[id2]:
            return 1
        else:
            return 0
    
    ##return 1 if it is in the decision class's lower approximate otherwise return 0
    def is_in_decision_lower(self,id1,attr_ids):##mf of id1 to [id1]d
        inf_mf = 1
        for id2 in range(self.X.shape[0]):
            if self.Y[id1] != self.Y[id2]:
                disim = 1 - self.relation_per_set(id1,id2,attr_ids)
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
                if self.Y[id1] != self.Y[id2]:
                    for attr_id in range(self.X.shape[1]):
                        if 1 - self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:# left side euqal to 0 or 1 and right side equal to 0 or 1, since it is a crisp rough set 
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
                    relation = self.relation_per_attr(id1,id2,attr)
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
        
    def relation_trim(self,by = 1):
        temp_tensor = []
        delete_row = []
        for attr in range(self.X.shape[1]):
            relation_mat = np.copy(self.relation_tensor[attr][by:,by:])
            delete_row.append(self.relation_tensor[attr][:by,:])
            temp_tensor.append(relation_mat)
        self.relation_tensor = temp_tensor
        return delete_row
#this method find the reduct of the initial dataset        
    def find_reduct(self):
##################################################################
# step 1: calculate the relation tensor 
        self.calculate_relation()
##################################################################
# step 2: calculate the membership function of each object to its decision class's lower approximation
        self.m_f = []
        for id1 in range(self.X.shape[0]):
            self.m_f.append(self.is_in_decision_lower(id1,list(range(self.X.shape[1]))))
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
        self.dis_red = set()

        for attr_id in red: #calculate the dis(reduct)
            value = self.dis_dict[attr_id]
            for pair in value:
                self.dis_red.add(pair)
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
     
    def fit(self, X, Y, X_var = []):
        self.Y = Y
        self.X = X
        if len(X_var) == 0 or len(X_var) != X.shape[1]:
            self.X_var = []
            for attr in range(self.X.shape[1]):
                self.X_var.append(np.var(self.X[:,attr]))
        else:
            self.X_var = X_var
        self.reduct_attr = self.find_reduct()

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
                        value = self.dis_dict[attr]
                        for pair in value:
                            dis_exclude_a.add(pair)
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
               
    def remove_first_object(self):
        drop_y = self.Y[0]
        self.Y = self.Y[1:]
        self.X = self.X[1:,:]
        self.m_f.remove(self.m_f[0])
        delete_row = self.relation_trim()
        for id1 in range(self.X.shape[0]):
            if self.Y[id1] == drop_y:
                pass
            else:
                sim = 0
                for attr in range(len(delete_row)):
                    sim = sim + delete_row[attr][0,id1+1]
                sim = sim/self.X.shape[1]
                if 1 - sim <= self.m_f[id1]:
                    self.m_f[id1] = self.is_in_decision_lower(id1,list(range(self.X.shape[1])))
                else:
                    pass
        for attr_id in self.dis_dict:
            pairs = self.dis_dict[attr_id]
            new_pairs = set()
            for pair in pairs:
                if 0 in pair:
                    pass
                else:
                    new_pairs.add((pair[0]-1,pair[1]-1))
            self.dis_dict[attr_id] = new_pairs
        new_pairs = set()
        for pair in self.dis_all:
            if 0 in pair:
                pass
            else:
                new_pairs.add((pair[0]-1,pair[1]-1))
        self.dis_all = new_pairs
        new_pairs = set()
        for pair in self.dis_red:
            if 0 in pair:
                pass
            else:
                new_pairs.add((pair[0]-1,pair[1]-1))
        self.dis_red = new_pairs      
        self.update_reduct()
    
    def remove_objects(self,by):
        drop_y = self.Y[:by]
        self.Y = self.Y[by:]
        self.X = self.X[by:,:]
        for i in range(by):
           self.m_f.remove(self.m_f[0])
        delete_row = self.relation_trim(by = by)
        for id1 in range(self.X.shape[0]):
            change = False
            for delete_id in range(by):
                if self.Y[id1] == drop_y[delete_id]:
                    pass
                else:
                    sim = 0
                    for attr in range(len(delete_row)):
                        sim = sim + delete_row[attr][delete_id,id1+by] 
                    sim = sim / self.X.shape[1]
                    if 1 - sim <= self.m_f[id1]:
                        change = True
                        break
                    else:
                        pass
            if change:
                self.m_f[id1] = self.is_in_decision_lower(id1,list(range(self.X.shape[1])))  
        for attr_id in self.dis_dict:
            pairs = self.dis_dict[attr_id]
            new_pairs = set()
            for pair in pairs:
                keep = True
                for id1 in pair:
                    if id1 < by:
                        keep = False
                if keep:
                   new_pairs.add((pair[0]-by,pair[1]-by)) 
            self.dis_dict[attr_id] = new_pairs
        new_pairs = set()
        for pair in self.dis_all:
            keep = True
            for id1 in pair:
                if id1 < by:
                    keep = False
            if keep:
                new_pairs.add((pair[0]-by,pair[1]-by)) 
        self.dis_all = new_pairs
        new_pairs = set()
        for pair in self.dis_red:
            keep = True
            for id1 in pair:
                if id1 < by:
                    keep = False
            if keep:
                new_pairs.add((pair[0]-by,pair[1]-by)) 
        self.dis_red = new_pairs
        self.update_reduct()
        
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
                sim = self.relation_per_attr(id1,-1,attr_id)
                self.relation_tensor[attr_id][id1,-1] = sim
                self.relation_tensor[attr_id][-1,id1] = sim
        #calculate the membership function of the new instance to its decision class's lower approximate
        self.m_f.append(self.is_in_decision_lower(-1,list(range(self.X.shape[1]))))#note index -1 means the last element
#       update the membership function of each instance to their decision class's lower approximate   
#       strategy for last row: we know m_f/d(x) <= 1 - R(x,y) for those y not agree with x in decision, here x denote the exist instance(not the new one)
#                              then if decision of x == decision of newX, then m_f/d(x) is unchanged
#                              if not agree on decision then m_f/d(x) = min(m_f/d(x), 1 - R(x,newX))        
        for id1 in range(self.X.shape[0]-1):
            if self.Y[id1] == newY:
                pass
            else:
                dism = 1 - self.relation_per_set(id1,-1,list(range(self.X.shape[1])))
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
                if self.Y[id1] != self.Y[-1]:
                    if 1- self.relation_tensor[attr_id][id1,-1] >= self.m_f[id1]:
                        self.dis_dict[attr_id].add((id1,self.X.shape[0]-1))
                        self.dis_all.add((id1,self.X.shape[0]-1))
                        if attr_id in self.reduct_attr:
                            self.dis_red.add((id1,self.X.shape[0]-1))
                    if 1- self.relation_tensor[attr_id][id1,-1] >= self.m_f[self.X.shape[0]-1]:
                        self.dis_dict[attr_id].add((self.X.shape[0]-1,id1))
                        self.dis_all.add((self.X.shape[0]-1,id1))
                        if attr_id in self.reduct_attr:
                            self.dis_red.add((self.X.shape[0]-1,id1))
# step4 update the discernibility set for all attribute this step update those not include newX, as newX come in, things change
#       since we update some membership degree when new instance comes in, there is a new threshold
#       and also remember we drop some pair when we calculate the reduct for initial dataset?
# for each instance pair (id1,id2):
#      for each attribute a:
#          if (id1,id2) not exist in dis(a) and d(id1) != d(id2) amd d(id1) != d(newX)
#                here d(id1) != d(newX) means m_f/d(id1) might change in the last step, we need to update
#                     d(id1) != d(id2) means they will be discerned by some attribute
#                and add the pair like what we do before for the initial data   
        for attr_id in self.dis_dict:
            for id1 in range(self.X.shape[0]-1):
                for id2 in range(self.X.shape[0]-1):
                    if not (id1,id2) in self.dis_dict[attr_id]:
                        if self.Y[id1] != self.Y[id2] and self.Y[id1] != self.Y[-1]:
                            if 1 - self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:
                                 self.dis_dict[attr_id].add((id1,id2))
                                 self.dis_all.add((id1,id2))
                                 if attr_id in self.reduct_attr:
                                      self.dis_red.add((id1,self.X.shape[0]-1))
        self.update_reduct()
                
    def update_group(self,newX,newY):
        self.Y = np.append(self.Y,newY)
        self.X = np.row_stack((self.X,newX))
        self.relation_expand(by = newX.shape[0])
        for attr_id in range(self.X.shape[1]):
            for id1 in range(self.X.shape[0] - newX.shape[0]):
                for id2 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
                    sim = self.relation_per_attr(id1,id2,attr_id)
                    self.relation_tensor[attr_id][id1,id2] = sim
                    self.relation_tensor[attr_id][id2,id1] = sim
        for attr_id in range(self.X.shape[1]):
            for id1 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0] - 1):
                for id2 in range(id1+1, self.X.shape[0]):
                    sim = self.relation_per_attr(id1,id2,attr_id)
                    self.relation_tensor[attr_id][id1,id2] = sim
                    self.relation_tensor[attr_id][id2,id1] = sim
        for id1 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
            self.m_f.append(self.is_in_decision_lower(id1,list(range(self.X.shape[1]))))
        for id1 in range(self.X.shape[0] - newX.shape[0]):
            for id2 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
                if self.Y[id1] == self.Y[id2]:
                    pass
                else:
                    dism = 1 - self.relation_per_set(id1,id2,list(range(self.X.shape[1])))
                    if self.m_f[id1] <= dism:
                        pass
                    else:
                        self.m_f[id1] = dism
        for id1 in range(self.X.shape[0] - newX.shape[0]):
            for id2 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
                for attr_id in self.dis_dict:
                    if self.Y[id1] != self.Y[id2]:
                        if 1- self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:
                            self.dis_dict[attr_id].add((id1,id2))
                            self.dis_all.add((id1,id2))
                            if attr_id in self.reduct_attr:
                                self.dis_red.add((id1,id2))
                        if 1- self.relation_tensor[attr_id][id1,id2] >= self.m_f[id2]:
                            self.dis_dict[attr_id].add((id2,id1))
                            self.dis_all.add((id2,id1))
                            if attr_id in self.reduct_attr:
                                self.dis_red.add((id2,id1))
        for attr_id in self.dis_dict:
            for id1 in range(self.X.shape[0] - newX.shape[0]):
                for id2 in range(self.X.shape[0] - newX.shape[0], self.X.shape[0]):
                    if self.Y[id1] != self.Y[id2]:
                        if 1- self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:
                            self.dis_dict[attr_id].add((id1,id2))
                            self.dis_all.add((id1,id2))
                            if attr_id in self.reduct_attr:
                                self.dis_red.add((id1,id2))
        for attr_id in self.dis_dict:
            for id1 in range(self.X.shape[0] - newX.shape[0]):
                for id2 in range(self.X.shape[0] - newX.shape[0]):
                    if not (id1,id2) in self.dis_dict[attr_id]:
                        if self.Y[id1] != self.Y[id2]:
                            if 1 - self.relation_tensor[attr_id][id1,id2] >= self.m_f[id1]:
                                 self.dis_dict[attr_id].add((id1,id2))
                                 self.dis_all.add((id1,id2))
                                 if attr_id in self.reduct_attr:
                                     self.dis_red.add((id1,id2))
        self.update_reduct()

   
class VIRS:## vote fuzzy rough set
    def __init__(self,batch_size):
        self.batch_size = batch_size
    def fit(self,X,Y, X_var = []):
        if X.shape[0]%self.batch_size == 0:
            self.child_num = int(X.shape[0]/self.batch_size)
        else:
            self.child_num = int(X.shape[0]/self.batch_size) + 1
        self.child_list = []
        if len(X_var) == 0 or len(X_var) != self.X.shape[1]:
            self.X_var = []
            for i in range(X.shape[1]):
                self.X_var.append(np.var(X[:,i]))
        else:
            self.X_var = X_var
        for i in range(self.child_num):
            tfrs = I_RS()
            tfrs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],self.X_var)
            self.child_list.append(tfrs)
    def update(self,newX,newY):
        if self.child_list[self.child_num-1].size() < self.batch_size:
            self.child_list[self.child_num-1].update(newX,newY)
        else:
            tfrs = I_RS()
            tfrs.fit(newX,newY,self.X_var)
            self.child_num = self.child_num + 1
            self.child_list.append(tfrs)
    def return_reduct(self):
        reduct_list = []
        for i in range(self.child_num):
            reduct_list.append(self.child_list[i].reduct_attr)
        return reduct_list
    ##voting style
    def predict(self,newX):
        pass
        
class TFVRS:## time fading vote fuzzy rough set
    def __init__(self,batch_size, fading_factor):
        self.batch_size = batch_size
        self.fading_factor = fading_factor
    def fit(self,X,Y, X_var = []):
        if X.shape[0]%self.batch_size == 0:
            self.child_num = int(X.shape[0]/self.batch_size)
        else:
            self.child_num = int(X.shape[0]/self.batch_size) + 1
        self.child_list = []
        if len(X_var) == 0 or len(X_var) != self.X.shape[1]:
            self.X_var = []
            for i in range(X.shape[1]):
                self.X_var.append(np.var(X[:,i]))
        else:
            self.X_var = X_var
        for i in range(self.child_num):
            tfrs = I_RS()
            tfrs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],self.X_var)
            self.child_list.append(tfrs)
    def update(self,newX,newY):
        if self.child_list[self.child_num-1].size() < self.batch_size:
            self.child_list[self.child_num-1].update(newX,newY)
        else:
            tfrs = I_RS()
            tfrs.fit(newX,newY,self.X_var)
            self.child_num = self.child_num + 1
            self.child_list.append(tfrs)
    def return_reduct(self):
        reduct_list = []
        for i in range(self.child_num):
            reduct_list.append(self.child_list[i].reduct_attr)
        return reduct_list
    def predict(self,newX):
       pass