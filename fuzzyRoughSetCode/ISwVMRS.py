import MRS as mrs
import numpy as np
class ISwMRS:## vote fuzzy rough set
    def __init__(self,batch_size,window_size):
        self.batch_size = batch_size
        self.window_size = window_size
    def fit(self,X,Y, cat = []):
        self.left_X = []
        self.left_Y = []
        self.child_list = []
        self.reduct_hist = []
        self.cat = cat
        if X.shape[0]%self.batch_size == 0:
            child_num = int(X.shape[0]/self.batch_size)
            start_batch = 0
            if child_num <= self.window_size:
                pass
            else:
                start_batch = child_num - self.window_size
            for i in range(start_batch,child_num):
                irs = mrs.MRS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],cat = self.cat)
                self.reduct_hist.append(irs.reduct_attr)
                self.child_list.append(irs)
        else:
            child_num = int(X.shape[0]/self.batch_size)
            start_batch = 0
            if child_num <= self.window_size:
                pass
            else:
                start_batch = child_num - self.window_size
            for i in range(start_batch,child_num):
                irs = mrs.MRS()
                irs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])],cat = self.cat)
                self.reduct_hist.append(irs.reduct_attr)
                self.child_list.append(irs)
            self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()
            self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
            
    def update(self,newX,newY):
        self.left_X.append(newX)
        self.left_Y.append(newY)
        if len(self.rule_lists) == self.batch_size:
            irs = mrs.MRS()
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.reduct_hist.append(irs.reduct_attr)
            self.child_list.append(irs)
            self.rule_combined = False
            self.left_X = []
            self.left_Y = []
            if len(self.child_list) > self.window_size:
                self.child_list.remove(self.child_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
            
    def update_group(self, newX, newY):
        num_to_fill_batch = self.batch_size-len(self.left_Y)
        if newX.shape[0] < num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
        elif newX.shape[0] == num_to_fill_batch:
            self.left_X.extend(newX.tolist())
            self.left_Y.extend(newY.tolist())
            irs = mrs.MRS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.reduct_hist.append(irs.reduct_attr)
            self.child_list.append(irs)
            self.left_X = []
            self.left_Y = []
            self.rule_combined = False
            if len(self.child_list) > self.window_size:
                self.child_list.remove(self.child_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
        else:
            to_fill_X = newX[:num_to_fill_batch]
            to_fill_Y = newY[:num_to_fill_batch]
            self.left_X.extend(to_fill_X.tolist())
            self.left_Y.extend(to_fill_Y.tolist())
            irs = mrs.MRS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y),cat = self.cat)
            self.reduct_hist.append(irs.reduct_attr)
            self.child_list.append(irs)
            self.left_X = []
            self.left_Y = []
            self.rule_combined = False
            if len(self.child_list) > self.window_size:
                self.child_list.remove(self.child_list[0])
                self.reduct_hist.remove(self.reduct_hist[0])
            newX = newX[num_to_fill_batch:]
            newY = newY[num_to_fill_batch:]
            self.update_group(newX,newY)
    def return_reduct(self):
        return self.reduct_hist
    def predict(self,newX):
        decision_dict = {}
        for child in self.child_list:
            decision,cover = child.predict(newX)
            if decision in decision_dict:
                decision_dict[decision][0] = decision_dict[decision][0] + 1
                decision_dict[decision][1] = decision_dict[decision][1] + cover
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

