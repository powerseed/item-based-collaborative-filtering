# This code generates the simulated hyperplane dataset used in many stream learning papers. In addition to outputting the dataset itself it also outputs the dimension weights over time. I used these dimension weights as a measure of true feature importance in a paper I discuss here: http://www.ccri.com/2014/10/30/calculating-feature-importance-in-data-streams-with-concept-drift-using-online-random-forest/

drift = function(start, numberToGenerate, magnitudeOfChange, probDirectionChange) {
  directions = rep(magnitudeOfChange, numberToGenerate - 1)
  for (i in 2:(numberToGenerate-1)) {
    if(!(runif(1) >= probDirectionChange)) {
      directions[i] = (directions[i-1] * -1)
    } else {
      directions[i] = (directions[i-1])
    }
  }
  builder = c(start)
  for (direction in directions) {
    start = start + direction
    builder = c(builder, start)
  }
  builder
}

hyperplane = function(num.to.generate, num.attributes, num.with.drift, mag.of.change, noise, prob.direction.change) {
  static.weights = runif(num.attributes - num.with.drift)
  static.weightz = do.call(rbind, lapply(1:num.to.generate, function(x) { static.weights}))
  dynamic.weights = do.call(cbind, lapply(runif(num.with.drift), function(start) { drift(start, num.to.generate, mag.of.change, prob.direction.change)}))
  weights = cbind(dynamic.weights, static.weightz)
  
  data = matrix(ncol=num.attributes, nrow=num.to.generate, runif(num.attributes * num.to.generate))
  data = as.data.frame(data)
  a.zero = apply(weights, 1, function(x) { .5*sum(x) })
  classes = apply(data * weights, 1, sum)  < a.zero
  # Do everything and then add noise
  c = sapply(classes, function(class) { if (runif(1) <= noise) { !class } else class })
  data$response = sapply(c, function(boo) { if(boo) {"class1"} else { "class2"}})
  list(data=data, weights=weights)
}

# This will generate 15 different datasets of 10000 observations
set.seed(1)
hyperplane.data = hyperplane(10000, 30, 5, 1.0, .05, .1)
write.csv(hyperplane.data$data, paste("hyperplane", 30, ".csv", sep=""), row.names=FALSE)


