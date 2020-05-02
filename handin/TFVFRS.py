import FRS as mrs
import numpy as np
# all the same as IVFRS but predition
class TFVFRS:## time fading vote fuzzy rough set
    def __init__(self,batch_size, fading_factor):
        self.batch_size = batch_size
        self.fading_factor = fading_factor
    def fit(self,X,Y, cat = []):
        self.left_X = []
        self.left_Y = []
        self.reduct_hist = []
        self.child_list = []
        self.cat = cat
        if X.shape[0]%self.batch_size == 0:
            child_num = int(X.shape[0]/self.batch_size)
            for i in range(child_num):
                irs = mrs.FRS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],cat = self.cat)
                self.child_list.append(irs)
        else:
            child_num = int(X.shape[0]/self.batch_size)
            for i in range(child_num):
                irs = mrs.FRS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],cat = self.cat)
                self.child_list.append(irs)
            self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()
            self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
            
    def update(self,newX,newY):
        self.left_X.append(newX)
        self.left_Y.append(newY)
        if len(self.all_rules) == self.batch_size:
            irs = mrs.FRS()
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.child_list.append(irs)
            self.left_X = []
            self.left_Y = []
            
    def update_group(self, newX, newY):
        num_to_fill_batch = self.batch_size-len(self.left_Y)
        if newX.shape[0] < num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
        elif newX.shape[0] == num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
            irs = mrs.FRS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.child_list.append(irs)
            self.left_X = []
            self.left_Y = []
        else:
            to_fill_X = newX[:num_to_fill_batch]
            to_fill_Y = newY[:num_to_fill_batch]
            self.left_X.extend(to_fill_X.tolist())
            self.left_Y.extend(to_fill_Y.tolist())
            irs = mrs.FRS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.child_list.append(irs)
            self.left_X = []
            self.left_Y = []
            newX = newX[num_to_fill_batch:]
            newY = newY[num_to_fill_batch:]
            self.update_group(newX,newY)
            
    def predict(self,newX):
        decision_dict = {}
        for i in range(len(self.child_list)):
            decision,cover = self.child_list[i].predict(newX)
            if decision in decision_dict:# we will fading the decision here
                decision_dict[decision][0] = decision_dict[decision][0] + self.fading_factor**(len(self.child_list)-(i+1))
                decision_dict[decision][1] = decision_dict[decision][1] + cover*self.fading_factor**(len(self.child_list)-(i+1))
            else:
                decision_dict[decision] = [1,cover]
        candidate_decision = []
        max_vote = 0
        for decision in decision_dict:
            if decision_dict[decision][0] > max_vote:
                candidate_decision = [[decision,decision_dict[decision][1]]]
                max_vote = decision_dict[decision][0]
            elif decision_dict[decision][0] == max_vote:
                candidate_decision.append([decision,decision_dict[decision][1]])
        if len(candidate_decision) == 1:
            return candidate_decision[0][0]
        else:
            max_cover = 0
            decision = None
            for candidate in candidate_decision:
                if candidate[1] > max_cover:
                    decision = candidate[0]
            return decision
