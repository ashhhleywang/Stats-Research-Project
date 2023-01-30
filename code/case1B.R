library(tidyr)
library(dplyr)
library(missForest)
library(doParallel)
library(randomForest)
library('fastDummies')
library(neuralnet)
library(keras)
load("../data/remData.Rdata")

r2_calc <- function(classifications, predictions){
  
  idx <- !is.na(predictions) & !is.na(classifications)
  classifications <- classifications[idx]
  predictions <- predictions[idx]
  
  1 - (sum((classifications-predictions)^2)/sum((classifications-mean(classifications))^2))
  
}

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)  

df = taksRem %>% select(-contains("_na"))

df2 = inner_join(df, grdXwalk, by = 'CAMPUS')
df2 = subset(df2,select = -c(COUNTY,GRDSPAN))
check = df2 %>% pivot_longer(cols = -c(CAMPUS,GRDTYPE), names_to = c("type","variable"),names_pattern = "(.+)([A-Z]\\d{2})",values_to = "value")

check = check %>% pivot_wider(names_from = type,values_from = value)

check2 = check %>% mutate(out = case_when(GRDTYPE %in% c("S", "B") & !is.na(outh) ~ outh,
                                          GRDTYPE %in% c("M", "E") ~ outm,
                                          GRDTYPE %in% c("S", "B") & is.na(outh) ~ outm))

tf = covsRem_noscale %>% select(CAMPUS,matches("_34$"))
ff = covsRem_noscale %>% select(CAMPUS,matches("_45$"))
fs = covsRem_noscale %>% select(CAMPUS,matches("_56$"))
ss = covsRem_noscale %>% select(CAMPUS,matches("_67$"))
se = covsRem_noscale %>% select(CAMPUS,matches("_78$"))

name_34 = tf[c(which(rowSums(is.na(tf)) == ncol(tf)-1)),1] # campus name whose column entries are all na == campus not existent in 34
name_45 = ff[c(which(rowSums(is.na(ff)) == ncol(ff)-1)),1]
name_56 = fs[c(which(rowSums(is.na(fs)) == ncol(fs)-1)),1]
name_67 = ss[c(which(rowSums(is.na(ss)) == ncol(ss)-1)),1]
name_78 = se[c(which(rowSums(is.na(se)) == ncol(se)-1)),1]



cv_fun <- function(X, y, K=10, cutoff=.85){
  N <- nrow(X)
  folds <- cut(sample(N),breaks=K,labels=FALSE)
  
  cv.out <- foreach(k = 1:K, .combine=rbind) %do% {
    
    xtrain = X[(folds !=k),]
    xtest = X[(folds ==k),]
    
    xtr_ori = xtrain %>% select(!starts_with('mis_'))
    xte_ori = xtest %>% select(!starts_with('mis_'))
    xtr_mis = xtrain %>% select(starts_with('mis_'))
    xte_mis = xtest %>% select(starts_with('mis_'))
    
    ind = apply(xtr_mis,2,function(X) !(all(X == 1)| all(X==0))) 
    mean(ind == TRUE) # 1  -> no col with all 1 / 0
    
    xtr_mis = xtr_mis[,ind]
    xte_mis = xte_mis[,ind]
    
    colmeans  = apply(xtr_mis,2,mean)
    colsds  = apply(xtr_mis,2,sd)
    
    X_scale <- scale(xtr_mis, center=colmeans, scale=colsds)
    
    svd_out <- svd(X_scale)
    prop_var = svd_out$d^2 / sum(svd_out$d^2)
    cum_var <- cumsum(prop_var)
    lvecs <- which(cum_var <= cutoff)
    
    a <- length(lvecs)
    print(a)
    top_lvecs <- svd_out$u[,1:a]
    
    U_train = svd_out$u[,1:a] #predictors for the training data
    
    xtr = cbind(xtr_ori,U_train)
    
    X_test_scaled = scale(xte_mis, center=colmeans, scale=colsds)
    
    U_test = X_test_scaled %*% svd_out$v[,1:a] %*% diag(1/svd_out$d[1:a])  #predictors for the test data
    
    xte = cbind(xte_ori,U_test)
    
    #rf_mod <- randomForest(xtr, y[(folds != k)],importance = TRUE)
    #rf_preds <- predict(rf_mod, newdata= xte)
    #rf_mse <- mean((rf_preds-y[(folds == k)])^2) 
    #rf_r2 <- r2_calc(y[(folds == k)],rf_preds) 
    

    train_cov = xtr
    train_target = y[(folds != k)]
    test_cov = xte
    test_target = y[(folds == k)]
    print('success')
    
    covs <- array(data = as.matrix(train_cov), dim = c(nrow(train_cov),1,ncol(train_cov)))
    response <- array(data = as.matrix(train_target), dim = c(nrow(train_cov), 1))
    covs_test = array(data = as.matrix(test_cov), dim = c(nrow(test_cov),1,ncol(test_cov)))
    response_test = array(data = as.matrix(test_target), dim = c(nrow(test_cov), 1))
    print('success')
    
    input_layer <- layer_input(shape = c(1,ncol(train_cov)))
    output_layer <- input_layer %>%
      #layer_masking(mask_value = 0) %>%
      #layer_dropout(rate = .5) %>%
      #layer_lstm(units = 64, return_sequences = F, dropout = .5, recurrent_dropout = .5) %>%
      layer_dense(units = 128, activation = "sigmoid") %>% 
      layer_dropout(rate = .5) %>%
      layer_dense(units = 1, activation = "linear")
    
    model <- keras_model(input_layer, output_layer)
    model %>% compile(
      loss = "mse",
      #metrics = "accuracy",
      optimizer = optimizer_adam()
    )
    
    early_stop <- callback_early_stopping(monitor = "val_loss", 
                                          #restore_best_weights = TRUE,
                                          patience = 10)
    
    history <- model %>% fit(
      x = covs,
      y = response,
      epochs = 1000,
      validation_split = 0.25,
      verbose = 0,
      callbacks = list(print_dot_callback,early_stop)
    )
    
    plot(history)
    
    pred <- model %>% predict(x = covs_test)
    nn1_mse <- mean((pred[,1,1] - response_test)^2)
    nn1_r2 <- r2_calc(response_test,pred[,1,1])
    
    
    input_layer <- layer_input(shape = c(1,ncol(train_cov)))
    output_layer <- input_layer %>%
      layer_dense(units = 128, activation = "sigmoid") %>% 
      #layer_masking(mask_value = 0) %>%
      layer_dropout(rate = .5) %>%
      #layer_lstm(units = 64, return_sequences = F, dropout = .5, recurrent_dropout = .5) %>%
      layer_dense(units = 128, activation = "sigmoid") %>% 
      layer_dropout(rate = .5) %>%
      layer_dense(units = 1, activation = "linear")
    
    model <- keras_model(input_layer, output_layer)
    model %>% compile(
      loss = "mse",
      #metrics = "accuracy",
      optimizer = optimizer_adam()
    )
    
    early_stop <- callback_early_stopping(monitor = "val_loss", 
                                          #restore_best_weights = TRUE,
                                          patience = 10)
    
    history <- model %>% fit(
      x = covs,
      y = response,
      epochs = 1000,
      validation_split = 0.25,
      verbose = 0,
      callbacks = list(print_dot_callback, early_stop)
    )
    
    plot(history)
    
    pred <- model %>% predict(x = covs_test)
    nn2_mse <- mean((pred[,1,1] - response_test)^2)
    nn2_r2 <- r2_calc(response_test,pred[,1,1])
    
    
    it.ob <- c(nn1_mse,nn1_r2,nn2_mse,nn2_r2)#,rf_mse,rf_r2)
    names(it.ob) <- c("nn1_mse","nn1_r2",'nn2_mse','nn2_r2')#,"rf_mse","rf_r2")
    
    it.ob
    
    # error_metric[k] <- mean((pred[,1] - response_test)^2) # 
    # r2_metric[k] = r2_calc(response_test,pred[,1]) #
  }
  
  return(cv.out)
}

## middle: replace with 0 ----
data_m = check2 %>% filter((!is.na(outm))&(is.na(outh)))
data_m = inner_join(data_m, covsRem_noscale, by = 'CAMPUS')
data_m = data_m %>% filter(variable == "A08")
data_m = merge(data_m, grdXwalk)
data_m = dummy_cols(data_m, select_columns = 'GRDSPAN') # 41 levels under GRDSPAN without na in outm
data_m = subset(data_m,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY)) # dont need to change out to outm becuz
# out is outm, without na as programmed

data_m$exist34 = ifelse(data_m$CAMPUS %in% name_34,1,0)
data_m$exist45 = ifelse(data_m$CAMPUS %in% name_45,1,0)
data_m$exist56 = ifelse(data_m$CAMPUS %in% name_56,1,0)
data_m$exist67 = ifelse(data_m$CAMPUS %in% name_67,1,0)
data_m$exist78 = ifelse(data_m$CAMPUS %in% name_78,1,0)

data_m <- data_m %>%
  mutate(across(everything(), ~ifelse(is.na(.x), 1, 0), .names="mis_{.col}"))

names(data_m) <- make.names(names(data_m))

data_m[,3:5145] = scale(data_m[,3:5145]) # 3-5145 original covariate
# replace na with 0
data_m[is.na(data_m)] <- 0 # 1394*10356

ind = apply(data_m,2,function(X) !(all(X == 1)| all(X==0)))
data_m = data_m[,ind] # 1394 * 10060

# check
#table(is.na(data_m))  # no na
#ind = apply(data_m,2,function(X) !(all(X == 1)| all(X==0)))
#mean(ind == TRUE) # 1  -> no col with all 1 / 0


#
X = data_m[,3:ncol(data_m)]
y = data_m[,2]
#
#
#
start_time <- Sys.time()
# #
middle_zero = cv_fun(X,y,K = 10)
#
end_time <- Sys.time()
end_time - start_time
#
save(middle_zero,file = 'case1B_middle_0.Rdata')
#
load('case1B_middle_0.Rdata')

# middle_zero
# cv_error = mean(middle_zero[,1])
# r2 = mean(middle_zero[,2])

# mean(middle_zero[,1])
# mean(middle_zero[,2])







## high 0 ----
data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)
data_h = dummy_cols(data_h, select_columns = 'GRDSPAN')
data_h = subset(data_h,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY))

data_h$exist34 = ifelse(data_h$CAMPUS %in% name_34,1,0)
data_h$exist45 = ifelse(data_h$CAMPUS %in% name_45,1,0)
data_h$exist56 = ifelse(data_h$CAMPUS %in% name_56,1,0)
data_h$exist67 = ifelse(data_h$CAMPUS %in% name_67,1,0)
data_h$exist78 = ifelse(data_h$CAMPUS %in% name_78,1,0)

data_h <- data_h %>%
  mutate(across(everything(), ~ifelse(is.na(.x), 1, 0), .names="mis_{.col}"))


names(data_h) <- make.names(names(data_h))
data_h[,3:5145] = scale(data_h[,3:5145])
# replace na with 0
data_h[is.na(data_h)] <- 0 # 1474* 10322
#data_h[0,]


ind = apply(data_h,2,function(X) !(all(X == 1)| all(X==0)))
data_h = data_h[,ind] # 1474 * 10256


X = data_h[,3:ncol(data_h)] 
y = data_h[,2] # 1474 * 1

start_time <- Sys.time()
#
high_zero = cv_fun(X,y,K = 10)

end_time <- Sys.time()
end_time - start_time

save(high_zero,file = 'case1B_high_0.Rdata')

load('case1B_high_0.Rdata')
# 
# high_zero
# cv_error = mean(high_zero[,1])
# r2 = mean(high_zero[,2])
# 
# mean(high_zero[,1])
# mean(high_zero[,2])

