import numpy as np
class I_TFRS:## incremental tolerance fuzzy tough set
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.X = None
        self.Y = None
        pass
    
    def dis(self):## return the discernibility
        pass
    
    def t_relation_per_set(self,id1,id2,attr_ids):
        t_norm = 1
        for attr_id in attr_ids:
            t_norm  = t_norm * self.t_relation_per_attr(id1,id2,attr_id)
        return t_norm
    
    def t_relation_per_attr(self,id1,id2,attr_id):## return the tolerance relation of x and y
        if attr_id == 'Y':
            disim = np.abs(self.Y[id1] - self.Y[id2])/np.abs(np.max(self.Y) - np.min(self.Y))
            return 1 - disim
        else:
            disim = np.abs(self.X[id1,attr_id] - self.X[id2, attr_id])/np.abs(np.max(self.X[:,attr_id]) - np.min(self.X[:,attr_id]))
            return 1 - disim
        
    def relation(self,id1,id2,attr_id): ## classical relation
        if attr_id == 'Y':
            return self.Y[id1] == self.Y[id2]
        else:
            return self.X[id1] == self.X[id2]
        
    def m_f_to_self_lowerapproximate(self,id1,attr_ids):##mf of id1 to [id1]d
        inf_mf = 1
        for id2 in range(self.X.shape[0]):
            if self.relation(id1,id2,'Y') != 1:
                disim = 1 - self.t_relation_per_set(id1,id2,attr_ids)
                if  disim < inf_mf:
                    inf_mf = disim
        return inf_mf
                
    def dis_matrix(self):
        dis_mat = []
        for id1 in range(self.X.shape[0]):
            dis_row = []
            for id2 in range(self.X.shape[0]):
                dis_set = []
                if self.relation(id1,id2,'Y') != 1:
                    for attr_id in range(self.X.shape[1]):
                        if 1 - self.t_relation_per_attr(id1,id2,attr_id) > self.m_f_to_self_lowerapproximate(id1,list(range(self.X.shape[1]))):
                            dis_set.append(attr_id)
                    dis_row.append(dis_set)
                else:
                    dis_row.append(dis_set)
            dis_mat.append(dis_row)
        return dis_mat
    
    def discernibility(self):
        dis_dict = {}
        for attr_id in list(range(self.X.shape[1])):
            dis_dict[attr_id] = set()
        dis_mat = self.dis_matrix()       
        for row in range(len(dis_mat)-1):
            for col in range(row+1,len(dis_mat[row])):
                for attr in dis_mat[row][col]:
                    dis_dict[attr].add((row,col))
        return dis_dict  
    def find_reduct(self):
        m_f = []
        for id1 in range(self.X.shape[0]):
            m_f.append(self.m_f_to_self_lowerapproximate(id1,list(range(self.X.shape[1]))))
        dis_dict = self.discernibility()
        dis_all = set()
        for key in dis_dict:
            for pair in dis_dict[key]:
                dis_all.add(pair)
        core = []
        for attr_id in list(range(self.X.shape[1])):
            dis_exclude_a = set()
            for key in dis_dict:
                if key == attr_id:
                    continue
                for pair in dis_dict[key]:
                    dis_exclude_a.add(pair)
            if dis_exclude_a != dis_all:
                core.append(attr_id)       
        red = core
        dis_red = set()

        for attr_id in red:
            value = dis_dict[attr_id]
            for pair in value:
                dis_red.add(pair)
        for attr_id in list(range(self.X.shape[1])):
            if not attr_id in red:
                dis_dict[attr_id] = dis_dict[attr_id].difference(dis_red)
        while dis_all != dis_red:
            max_size = -1
            candidate = -1
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    if len(dis_dict[attr_id]) > max_size:
                        max_size = len(dis_dict[attr_id])
                        candidate = attr_id
            red.append(candidate)
            dis_red = dis_red.union(dis_dict[candidate])
            for attr_id in list(range(self.X.shape[1])):
                if not attr_id in red:
                    dis_dict[attr_id] = dis_dict[attr_id].difference(dis_red)
        return red
    def fit(self,decision_table, decision_col_name):
        self.Y = decision_table[decision_col_name].values
        self.X = decision_table.drop([decision_col_name],axis = 1).values
        self.reduct_attr = self.find_reduct()

########################################################################################################################        
    def m_f(self,id1,decision):
        if self.Y[id1] == decision:
            return 1
        else:
            return 0

    def m_f_to_positive_region(self,id1,attr_set):
        mf = 0
        for decision in np.unique(self.Y):
            inf_mf = np.inf
            for id2 in range(self.X.shape[0]):
              t_norm = max(1-self.t_relation_per_set(id1,id2,attr_set),self.m_f(id2,decision))  
              if t_norm < inf_mf:
                  inf_mf = t_norm
            if inf_mf > mf:
                mf = inf_mf               
        return mf
    
    def predictibility(self,attr_set):
        if len(attr_set) == 0:
            return 0
        elif len(attr_set) == self.X.shape[1]:
            return 1
        b_sum = 0
        a_sum = 0
        for id1 in range(self.X.shape[0]):
            b_sum = b_sum + self.m_f_to_positive_region(id1,attr_set)
            a_sum = a_sum + self.m_f_to_positive_region(id1,list(range(self.X.shape[1])))
        return b_sum / a_sum
    
    def quick_reduct(self):
        reduct = []
        while self.predictibility(reduct) < self.predictibility(list(range(self.X.shape[1])))*self.tolerance:
            temp = reduct.copy()
            for attr in range(self.X.shape[1]):
                if attr in reduct:
                    continue
                else:
                    copy = reduct.copy()
                    copy.append(attr)
                    if self.predictibility(copy) > self.predictibility(temp):
                        temp = copy
            reduct = temp
            print(reduct)
        return reduct
######################################################################################################################                   
            
            
import pandas as pd
df = pd.read_csv('breast_cancer_diagnosis.csv')    
df = df.drop(['id'],axis = 1)
tfrs = I_TFRS(0.8)
tfrs.fit(df[:100],'diagnosis')                    
        