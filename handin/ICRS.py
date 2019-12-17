import numpy as np
import RS as rs
#class ICRS(incremental cumulative rough set)
class ICRS:
     def __init__(self,batch_size):
        self.batch_size = batch_size
     # this for initialize the model   
     def fit(self,X,Y):
        self.left_X = []# the left data that could not fit in a batch, will wait until more data come in
        self.left_Y = []
        self.all_rule = {}# the agrregated rules
        self.reduct_hist = []# the reduct history        
        child_num = int(X.shape[0]/self.batch_size)
        for i in range(child_num):
            irs = rs.I_RS()
            irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])])
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
        ###########################################possible rule, not used, here is one place you could extend################
            #self.aggregate(irs.possible_rule) 
        ######################################################################################################################
            if X.shape[0]%self.batch_size != 0:# if there is data left and not enough for a batch
                self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()# add the left data
                self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
     
    # this for update the model by one single instance       
     def update(self,newX,newY):
        self.left_X.append(newX)# simiply append it to the list
        self.left_Y.append(newY)
        if len(self.all_rules) == self.batch_size:# if there is enough data to generate a batch
            irs = rs.I_RS()
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
        elif newX.shape[0] == num_to_fill_batch:# if exactly equal to number to fill
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
        ###########################################possible rule, not used, here is one place you could extend################
            #self.aggregate(irs.possible_rule) 
        ######################################################################################################################
            self.left_X = []
            self.left_Y = []
        else:# if more than number to fill
            to_fill_X = newX[:num_to_fill_batch]
            to_fill_Y = newY[:num_to_fill_batch]
            self.left_X.extend(to_fill_X.tolist())
            self.left_Y.extend(to_fill_Y.tolist())
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
            self.reduct_hist.append(irs.reduct_attr)
            self.aggregate(irs.fix_rule)       
        ###########################################possible rule, not used, here is one place you could extend################
            #self.aggregate(irs.possible_rule) 
        ######################################################################################################################
            self.left_X = []
            self.left_Y = []
            newX = newX[num_to_fill_batch:]# the left data go to another round
            newY = newY[num_to_fill_batch:]
            self.update_group(newX,newY)
     
     def return_reduct(self):
        return self.reduct_hist
    
     def aggregate(self,rules):
    # for each fix rule: note the rule with format [condition, decision, support, confident]
        for rule in rules:
            if (rule[0],rule[1]) in self.all_rule:# if the rule exist, aggregate them
                current_coverage = self.all_rule[(rule[0],rule[1])] # coverage with format [support, confidence]
                # new_support = current_support + rule support
                # new_confidence = (current_supporT*current_confidence + rule_support*rule_confidence)/new_support
                new_coverage = (current_coverage[0]+rule[2],(current_coverage[0]*current_coverage[1]+rule[2]*rule[3])/(rule[2]+current_coverage[0]))
                self.all_rule[(rule[0],rule[1])] = new_coverage
            else:# if not exist, initialize it
                self.all_rule[(rule[0],rule[1])] = (rule[2],rule[3])
     #predition on target object           
     def predict(self,newX):
        match_decision = {}
        for rule in self.all_rule:# iterate through all the rule format:[condition, decision]:[support, confidence]
           condition = rule[0]
           match = True
           for pair in condition:
               if newX[pair[0]] != pair[1]:
                   match = False
                   break
           if match:
               if rule[1] in match_decision: #formate decision:[support, confidence]
                   match_decision[rule[1]][0] = match_decision[rule[1]][0]+self.all_rule[rule][0]#add support
                   match_decision[rule[1]][1] = match_decision[rule[1]][1]+self.all_rule[rule][1]#add confidence!!! might not right, but we don care now since we ignore possible rule, all fule with confidence 1
               else:
                   match_decision[rule[1]] = [self.all_rule[rule][0],self.all_rule[rule][1]]
        if len(match_decision) == 1:#only one decision, return it
           for decision in match_decision:
               return decision
        elif len(match_decision) > 1: #more than one decision, find the one with max support
           max_cover = 0
           final_decision = None
           for decision in match_decision:
               if match_decision[decision][0] > max_cover:
                   max_cover = match_decision[decision][0]
                   final_decision = decision
           return final_decision
        elif len(match_decision) == 0:# no match, we want our prediction are with evidence, so we return unknown, as a result, all of the prediction is based on mined knowledge instead of guess
           return 'Unknown'
