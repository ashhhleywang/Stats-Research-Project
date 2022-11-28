cv.out <- foreach(k = 1:K, .combine=rbind) %do% {
  #fit model on the training data for this fold (hold out the fold)
  rf <- randomForest(X[(folds != k),], y[(folds != k)],importance = TRUE)
  
  imp = sort(rf$importance[,1]/sum(rf$importance[,1]),decreasing = T)[1:20]
  names(imp) #top 20 important var
  
  # fit neural network
  train_cov = X[(folds != k),c(names(imp))] 
  train_target = y[(folds != k)]
  test_cov = X[(folds == k),c(names(imp))]
  test_target = y[(folds == k)]
  
  covs <- array(data = as.matrix(train_cov), dim = c(nrow(train_cov),1,ncol(train_cov)))
  response <- array(data = as.matrix(train_target), dim = c(nrow(train_cov), 1))
  covs_test = array(data = as.matrix(test_cov), dim = c(nrow(test_cov),1,ncol(test_cov)))
  response_test = array(data = as.matrix(test_target), dim = c(nrow(test_cov), 1))
  
  input_layer <- layer_input(shape = c(1,ncol(X)))
  output_layer <- input_layer %>%
    layer_masking(mask_value = 0) %>%
    layer_dropout(rate = .5) %>%
    layer_lstm(units = 64, return_sequences = F, dropout = .5, recurrent_dropout = .5) %>%
    layer_dropout(rate = .5) %>%
    layer_dense(units = 1, activation = "linear")
  
  
  model <- keras_model(input_layer, output_layer)
  
  #apply the model to the held out fold data (test)
  pred <- predict(rf, newdata = X[(folds == k),])
  
  #calculate the MSE and save for this fold
  #replace with whatever error metric you want and add more
  error_metric <- mean((pred - y[(folds == k)])^2)
  r2_metric <- r2_calc(y[(folds == k)],pred)
  
  it.ob <- c(error_metric, r2_metric)
  names(it.ob) <- c("mse","r2")
  
  it.ob
  
}


cv_error = mean(cv.out$mse)
r2 = mean(cv.out$r2)
end_time <- Sys.time()