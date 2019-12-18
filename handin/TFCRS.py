import numpy as np
import RS as rs
class TFCRS:
     def __init__(self,batch_size,rate):
        self.batch_size = batch_size
        self.rate = rate
    
     def fit(self,X,Y):
        self.left_X = []
        self.left_Y = []
        self.all_rule = {}
        self.reduct_hist = []    
        child_num = int(X.shape[0]/self.batch_size)
        for i in range(child_num):
            self.fade()# fade the older data
            irs = rs.RS()
            irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])])
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
    ###########################################possible rule, not used, here is one place you could extend################
        #self.aggregate(irs.possible_rule) 
    ######################################################################################################################
        if X.shape[0]%self.batch_size != 0:    
            self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()
            self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
        #apply time fading factor to all the rule
     def fade(self):
         for rule in self.all_rule:
             first = self.all_rule[rule][0]*self.rate
             second = self.all_rule[rule][1]#should confidence also fade?
             self.all_rule[rule] = (first,second)
     
     def update(self,newX,newY):
        self.left_X.append(newX)
        self.left_Y.append(newY)
        if len(self.all_rules) == self.batch_size:
            self.fade()# fade the older data
            irs = rs.RS()
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
    ###########################################possible rule, not used, here is one place you could extend################
        #self.aggregate(irs.possible_rule) 
    ######################################################################################################################
            self.left_X = []
            self.left_Y = []
            
     def update_group(self, newX, newY):
        num_to_fill_batch = self.batch_size-len(self.left_Y)
        if newX.shape[0] < num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
        elif newX.shape[0] == num_to_fill_batch:
            self.fade()# fade the older data
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
            irs = rs.RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
    ###########################################possible rule, not used, here is one place you could extend################
        #self.aggregate(irs.possible_rule) 
    ######################################################################################################################
            self.left_X = []
            self.left_Y = []
        else:
            to_fill_X = newX[:num_to_fill_batch]
            to_fill_Y = newY[:num_to_fill_batch]
            self.left_X.extend(to_fill_X.tolist())
            self.left_Y.extend(to_fill_Y.tolist())
            self.fade()# fade the older data
            irs = rs.RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
    ###########################################possible rule, not used, here is one place you could extend################
        #self.aggregate(irs.possible_rule) 
    ######################################################################################################################
            self.left_X = []
            self.left_Y = []
            newX = newX[num_to_fill_batch:]
            newY = newY[num_to_fill_batch:]
            self.update_group(newX,newY)
     def return_reduct(self):
        return self.reduct_hist
    
     def aggregate(self,rules):
    # for each fix rule: note the rule with format [condition, decision, support, confident]
        for rule in rules:
            if (rule[0],rule[1]) in self.all_rule:# if the rule exist, aggregate them
                current_coverage = self.all_rule[(rule[0],rule[1])] # coverage with format [support, confidence]
                # comment: new_support = current_support + rule support
                # comment: new_confidence = (current_supporT*current_confidence + rule_support*rule_confidence)/new_support
                new_coverage = (current_coverage[0]+rule[2],(current_coverage[0]*current_coverage[1]+rule[2]*rule[3])/(rule[2]+current_coverage[0]))
                self.all_rule[(rule[0],rule[1])] = new_coverage
            else:# if not exist, initialize it
                self.all_rule[(rule[0],rule[1])] = (rule[2],rule[3])  
     # identical to ICRS           
     def predict(self,newX):
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
