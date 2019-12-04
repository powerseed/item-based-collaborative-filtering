import IRS1 as rs
class TFVRS:## time fading vote fuzzy rough set
    def __init__(self,batch_size, fading_factor):
        self.batch_size = batch_size
        self.fading_factor = fading_factor
    def fit(self,X,Y):
        if X.shape[0]%self.batch_size == 0:
            self.child_num = int(X.shape[0]/self.batch_size)
        else:
            self.child_num = int(X.shape[0]/self.batch_size) + 1
        self.child_list = []
        for i in range(self.child_num):
            tfrs = rs.I_RS()
            tfrs.fit(X[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0]),:],Y[i*self.batch_size:min((i+1)*self.batch_size,Y.shape[0])])
            self.child_list.append(tfrs)
    def update(self,newX,newY):
        if self.child_list[self.child_num-1].size() < self.batch_size:
            self.child_list[self.child_num-1].update(newX,newY)
        else:
            tfrs = rs.I_RS()
            tfrs.fit(newX,newY)
            self.child_num = self.child_num + 1
            self.child_list.append(tfrs)
    def update_group(self, newX, newY):
        if self.child_list[self.child_num-1].size() + newX.shape[0] <= self.batch_size:
            self.child_list[self.child_num-1].update(newX,newY)
        else:
            first_half = self.batch_size - self.child_list[self.child_num-1].size()
            if first_half != 0:
                self.child_list[self.child_num-1].update_group(newX[:first_half],newY[:first_half])
                newX = newX[first_half:]
                newY = newY[first_half:]
            while newX.shape[0] > self.batch_size:
                self.child_num = self.child_num + 1
                tfrs = rs.I_RS()
                tfrs.fit(newX[:self.batch_size],newY[:self.batch_size])
                self.child_list.append(tfrs)
                newX = newX[self.batch_size:]
                newY = newY[self.batch_size:]
            self.child_num = self.child_num + 1
            tfrs = rs.I_RS()
            tfrs.fit(newX,newY)
            self.child_list.append(tfrs)
    def return_reduct(self):
        reduct_list = []
        for i in range(self.child_num):
            reduct_list.append(self.child_list[i].reduct_attr)
        return reduct_list
    def predict(self,newX):
       predictions = {}
       for i in range(self.child_num):
            prediction,mf = self.child_list[i].predict(newX)
            if prediction != 'Unknown':
                if prediction in predictions:
                    predictions[prediction][0] = predictions[prediction][0] + self.fading_factor**(self.child_num-(i+1))
                    predictions[prediction][1] = predictions[prediction][1] + mf * self.fading_factor**(self.child_num-(i+1))
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
