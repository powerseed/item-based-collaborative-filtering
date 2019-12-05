import RS as rs
import numpy as np
class ISwVRS:## vote fuzzy rough set
    def __init__(self,batch_size,window_size):
        self.batch_size = batch_size
        self.window_size = window_size
        
    def fit(self,X,Y):
        self.left_X = []
        self.left_Y = []
        self.child_list = []
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
                self.child_list.append(irs)
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
                self.child_list.append(irs)
            self.left_X = X[child_num*self.batch_size:X.shape[0],:].tolist()
            self.left_Y = Y[child_num*self.batch_size:Y.shape[0]].tolist()
            
    def update(self,newX,newY):
        self.left_X.append(newX)
        self.left_Y.append(newY)
        if len(self.rule_lists) == self.batch_size:
            irs = rs.I_RS()
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
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
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
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
            irs = rs.I_RS()
            #print((len(self.left_X),len(self.left_Y)))
            irs.fit(np.array(self.left_X),np.array(self.left_Y))
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
    ##voting style
    def predict(self,newX):
        predictions = {}
        for i in range(len(self.child_list)):
            prediction,mf = self.child_list[i].predict(newX)
            if prediction != 'Unknown':
                if prediction in predictions:
                    predictions[prediction][0] = predictions[prediction][0] + 1
                    predictions[prediction][1] = predictions[prediction][1] + mf
                else:
                    predictions[prediction] = [1,mf]
        max_key = []
        max_support = 0
        for key in predictions:
            if predictions[key][0] > max_support:
                max_key = [key]
                max_support = predictions[key][0]
            elif predictions[key][0] == max_support:
                max_key.append(key)
            else:
                pass
        if len(max_key) == 1:
            return max_key[0]
        elif len(max_key) == 0:
            return 'Unknown'
        else:
            max_mf = 0
            prediction = None
            for i in range(len(max_key)):
                if predictions[max_key[i]][1] > max_mf:
                    max_mf = predictions[max_key[i]][1]
                    prediction = max_key[i]
            return prediction  
