import numpy as np
from sklearn.cluster import KMeans
class FRS:## incremental tolerance fuzzy tough set
    def __init__(self):## we did not set them as fix, you could do that
        self.max_rule_per_decision = 10
        self.rule_proportion = 0.1
    
    ##return the similarity of two object by the attribute id in attr_ids
    #input id1: the id of the first object
    #      id2: the second object
    #      attr_ids: the id of the attributes
    # strtegy: since this is a fuzzy rough set, the similarity aggregation(defined as t_norm in fuzzy set theory) that we could choos from include min, max, and product
    #          the one we pick is the product. So this method just mutiply up the similarity among all the attribute
    # output: tnorm: is the similarity of two object in tese attribute
    def t_relation_per_set(self,id1,id2,attr_ids):
        t_norm = 1
        for attr_id in attr_ids:
            t_norm  = min(t_norm,self.relation_tensor[attr_id][id1,id2])
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
        if attr_id in self.cat:
            if self.X[id1,attr_id] == self.X[id2,attr_id]:
                return 1
            else:
                return 0
        relation = np.exp(-((self.X[id1,attr_id] - self.X[id2,attr_id])**2)/(2*self.X_var[attr_id]))
        if np.isnan(relation):##just in case variance is 0, then it make sense that they are the same 
            relation = 1
        return relation
    
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
            if self.Y[id1] != self.Y[id2]:
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
                if self.Y[id1] != self.Y[id2]:
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
        dis_dict = self.discernibility()
        dis_all = set()## since it is a set, add duplicate to it will get ignore
        for key in dis_dict:
            for pair in dis_dict[key]:
                dis_all.add(pair)
#################################################################
# step 4: calculate the core of the initial dataset
# we do this by calculate the discernibility of A - a for all a belong A (A denote the combination of all attribute and a is the attributes in A)
# then if there is an 'a' such that dis(A - a) != dis(A), a should be the in the core
        core = []
        for attr_id in list(range(self.X.shape[1])):
            dis_exclude_a = set() #calculate the discernibility of A - a for all a belong A
            for key in dis_dict:
                if key == attr_id:
                    continue
                for pair in dis_dict[key]: # pair is (id1,id2) tuple, use tuple as it could be hashed
                    dis_exclude_a.add(pair)
            if dis_exclude_a != dis_all:#dis(A - a) != dis(A)
                core.append(attr_id) #a should be the in the core   
#################################################################
# step 5, calculate the reduct
# initialize the reduct with core, calculate the dis(reduct) and for the discenibility set of each attribute, exclude those in the dis(reduct) 
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)
        red = core#initialize the reduct with core
        dis_red = set()

        for attr_id in red: #calculate the dis(reduct)
            value = dis_dict[attr_id]
            for pair in value:
                dis_red.add(pair)
        #for the discenibility set of each attribute, exclude those in the dis(reduct)  
        copy_dis_dict = dis_dict.copy()
        for attr_id in list(range(self.X.shape[1])):
            if not attr_id in red:
                copy_dis_dict[attr_id] = copy_dis_dict[attr_id].difference(dis_red)
# then iteratively select the attibute with highest discernibility(that is the discernibility set have more element)
# until we get the reduct (hill climbing)                
        while dis_all != dis_red:
            max_size = -1
            candidate = -1
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    if len(copy_dis_dict[attr_id]) > max_size:
                        max_size = len(copy_dis_dict[attr_id])
                        candidate = attr_id
            red.append(candidate)
            dis_red = dis_red.union(copy_dis_dict[candidate])
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    copy_dis_dict[attr_id] = copy_dis_dict[attr_id].difference(dis_red)
        self.relation_tensor = None
        return red
## input:
#       X: conditional attribute and values
#       Y: decision value
#       cat: index of catorgories attribute: !!!but not finish yet, does use it!!!
#       var: varriance of each attribute
#procedure:
#       firstly calculate the variance of each attribute: since we are using gaussian similarity, we need it
#       then find the reduct of the rough set system
#       finally use the reduct to find the rule objects
    def fit(self, X, Y, cat = [], X_var = []):
        self.Y = Y
        self.X = X
        self.cat = cat
        if len(X_var) == 0 or len(X_var) != X.shape[1]:
            self.X_var = []
            for attr in range(self.X.shape[1]):
                if attr in cat:
                    self.X_var.append(0)
                else:
                    self.X_var.append(np.var(self.X[:,attr]))
        else:
            self.X_var = X_var
        
        self.reduct_attr = self.find_reduct()
        self.minimum_rule_finding()
# this routine predict the decision of the newX target object
# return the predicted decision as well as newX's estimation menbership degree to this decision
# the estimation of membership degree will be used in the three proposed model
# when you test on this set only, remenber use prediction[0]
#general idea:
#       for each decision, calculate the membership degree of the target object to the decision's upper approximate and lower approximate
#       pick the decision with the maximum sum of these two degree
#       detail of definition of membership degree to approximation pls refer to report
    def predict(self,newX):
        best_decision = -1
        max_mf = 0
        sim_dict = {}# a dictionary to hold similarity
        # for efficiency, firstly calculate all the similarity and store them in sim_dict
        for rules_decision in self.all_rule:
            rules = self.all_rule[rules_decision]
            for rule in rules:
                sim = 1
                for attr in self.reduct_attr:# only comparing the reduct attribute, more ttolerance, more efficient
                    if attr in self.cat:# our model now could not handle catorgorical attribute, but put here for later
                        if newX[attr] == rule[attr]:
                            sim = min(sim,1)
                        else:
                            sim = 0
                    else:
                        relation = np.exp(-(newX[attr] - rule[attr])**2/(2*self.X_var[attr])) #guassian similarity
                        if np.isnan(relation):##just in case variance is 0, then it make sense that they are the same 
                            relation = 1
                        sim  = min(sim,relation)# t-norm
                sim_dict[tuple(rule)] = sim   # record the similarity, the key is the rule object     
        for decision in np.unique(self.Y):# for each decision
            m_f_to_low = 0 # mebership degree to current decision's lower approximate
            m_f_to_up = 0 # mebership degree to current decision's upper approximate
            for rules_decision in self.all_rule:# iterate through the rules, each rule set contain all the rules for a single decision
                if rules_decision != decision: # if decision is not agree, we will use it to find lower approximation
                    rules = self.all_rule[rules_decision] # here is all the rule objects
                    for rule in rules:# iterate through the rule
                        sim = sim_dict[tuple(rule)]
                        if m_f_to_low < 1 - sim:
                            m_f_to_low = 1 - sim
                if rules_decision == decision: # if decision is agree, we will use it to find upper approximation
                    rules = self.all_rule[rules_decision]
                    for rule in rules:
                        sim = sim_dict[tuple(rule)]
                        if m_f_to_up < sim:
                            m_f_to_up= sim
            mf = m_f_to_low +m_f_to_up# mf will be the average of low and up, but here we think just the sum will also be fine
            if mf > max_mf:
                max_mf = mf
                best_decision = decision
        return best_decision,max_mf
   # using K means to find the ruling object 
    def minimum_rule_finding(self):
        self.all_rule = {}     
        for decision in np.unique(self.Y):#for each decision
            index_d = np.where(self.Y == decision)[0]  # the index in X that with such decision
            U_X = self.X[index_d]# get those object with current decision
            centers = U_X # initialize the center by all the object with current decision
            if len(self.cat) == 0: # if there is no catergorical attribute, use k means to find the center as rule, otherwise, use all objects as rule
                if U_X.shape[0] > self.max_rule_per_decision: #
                        kmodel = KMeans(n_clusters = max(int(U_X.shape[0]*self.rule_proportion),self.max_rule_per_decision), n_jobs = 8)
                        kmodel.fit(U_X)
                        centers = kmodel.cluster_centers_
            self.all_rule[decision] = centers# mark those center as rule
         
   

    

