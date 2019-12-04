import numpy as np
import IRS1 as rs
class ISwCRS:
     def __init__(self,batch_size,window_size):
        self.batch_size = batch_size
        self.window_size = window_size
        self.rule_combined = False
    
     def fit(self,X,Y):
        self.left_X = []
        self.left_Y = []
        self.rule_list = []
        self.reduct_hist = []
        if X.shape[0]%self.batch_size == 0:
            child_num = int(X.shape[0]/self.batch_size)
            start_batch = 0
            if child_num <= self.window_size:
                pass
            else:
                start_batch = child_num - self.window_size
            for i in range(start_batch,child_num):
                irs = rs.I_RS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])])
                self.reduct_hist.append(irs.reduct_attr)
                self.rule_list.append(irs.fix_rule)
        else:
            child_num = int(X.shape[0]/self.batch_size)
            start_batch = 0
            if child_num <= self.window_size:
                pass
            else:
                start_batch = child_num - self.window_size
            for i in range(start_batch,child_num):
                irs = rs.I_RS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])])
                self.reduct_hist.append(irs.reduct_attr)
                self.rule_list.append(irs.fix_rule)
            self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()
            self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
            
     def update(self,newX,newY):
        self.left_X.append(newX)
        self.left_Y.append(newY)
        if len(self.rule_lists) == self.batch_size:
            irs = rs.I_RS()
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.rule_list.append(irs.fix_rule)
            self.rule_combined = False
            self.left_X = []
            self.left_Y = []
            if len(self.rule_list) > self.window_size:
                self.rule_list.remove(self.rule_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
            
     def update_group(self, newX, newY):
        num_to_fill_batch = self.batch_size-len(self.left_Y)
        if newX.shape[0] < num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
        elif newX.shape[0] == num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.rule_list.append(irs.fix_rule)
            self.left_X = []
            self.left_Y = []
            self.rule_combined = False
            if len(self.rule_list) > self.window_size:
                self.rule_list.remove(self.rule_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
        else:
            to_fill_X = newX[:num_to_fill_batch]
            to_fill_Y = newY[:num_to_fill_batch]
            self.left_X.extend(to_fill_X.tolist())
            self.left_Y.extend(to_fill_Y.tolist())
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.rule_list.append(irs.fix_rule)
            self.left_X = []
            self.left_Y = []
            self.rule_combined = False
            if len(self.rule_list) > self.window_size:
                self.rule_list.remove(self.rule_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
            newX = newX[num_to_fill_batch:]
            newY = newY[num_to_fill_batch:]
            self.update_group(newX,newY)
     def return_reduct(self):
        return self.reduct_hist
    
     def combine_rule(self):
         self.all_rule = {}
         if not self.rule_combined:
             for rules in self.rule_list:
                 for rule in rules:
                    if (rule[0],rule[1]) in self.all_rule:
                        current_coverage = self.all_rule[(rule[0],rule[1])]
                        new_coverage = (current_coverage[0]+rule[2],(current_coverage[0]*current_coverage[1]+rule[2]*rule[3])/(rule[2]+current_coverage[0]))
                        self.all_rule[(rule[0],rule[1])] = new_coverage
                    else:
                        self.all_rule[(rule[0],rule[1])] = (rule[2],rule[3])
             self.rule_combined = True
                    
     def predict(self,newX):
        self.combine_rule()
        match_decision = {}
        for rule in self.all_rule:
           condition = rule[0]
           match = True
           for pair in condition:
               if newX[pair[0]] != pair[1]:
                   match = False
                   break
           if match:
               if rule[1] in match_decision:
                   match_decision[rule[1]][0] = match_decision[rule[1]][0]+self.all_rule[rule][0]
                   match_decision[rule[1]][1] = match_decision[rule[1]][1]+self.all_rule[rule][1]
               else:
                   match_decision[rule[1]] = [self.all_rule[rule][0],self.all_rule[rule][1]]
        if len(match_decision) == 1:
           for decision in match_decision:
               return decision
        elif len(match_decision) > 1:
           max_cover = 0
           final_decision = None
           for decision in match_decision:
               if match_decision[decision][0] > max_cover:
                   max_cover = match_decision[decision][0]
                   final_decision = decision
           return final_decision
        elif len(match_decision) == 0:
           return 'Unknown'

