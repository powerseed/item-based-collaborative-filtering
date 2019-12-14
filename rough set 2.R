library(RoughSets)
library(arules)
df = discretizeDF(read.csv('movingRBF.csv')[,0:10], default = list(method = "cluster", breaks = 10,include.lowest = TRUE))
target = sapply(read.csv('movingRBF.csv')[,11],as.character)
df = cbind(df,target)
i = 2000
correct = 0
while (i < 200000){
  decision.table <- SF.asDecisionTable(dataset = df[0:i,], decision.attr = 11)
  red.rst <- FS.feature.subset.computation(decision.table,method="quickreduct.rst")
  rules <- RI.indiscernibilityBasedRules.RST(decision.table,red.rst)
  test = SF.asDecisionTable(dataset = df[i+1,], decision.attr = 11)
  pred = predict(rules,test)
  real = target[i+1]
  correct = correct + sum(pred == real)
  i = i + 1
}

