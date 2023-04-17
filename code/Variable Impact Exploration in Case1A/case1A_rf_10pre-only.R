library(tidyr) 
library(dplyr)
library(missForest)
library(doParallel)
library(randomForest)
library('fastDummies')
library(neuralnet)
library(keras)
library(stringr)
library(readr)

load("../../data/temp/remData.Rdata")

#check3 = covsRem_noscale %>% select(starts_with('pre'))


r2_calc <- function(classifications, predictions){
  
  idx <- !is.na(predictions) & !is.na(classifications)
  classifications <- classifications[idx]
  predictions <- predictions[idx]
  
  1 - (sum((classifications-predictions)^2)/sum((classifications-mean(classifications))^2))
  
}



df = taksRem %>% select(-contains("_na"))


df2 = inner_join(df, grdXwalk, by = 'CAMPUS')
df2 = subset(df2,select = -c(COUNTY,GRDSPAN))
check = df2 %>% pivot_longer(cols = -c(CAMPUS,GRDTYPE), names_to = c("type","variable"),names_pattern = "(.+)([A-Z]\\d{2})",values_to = "value")

check = check %>% pivot_wider(names_from = type,values_from = value)

check2 = check %>% mutate(out = case_when(GRDTYPE %in% c("S", "B") & !is.na(outh) ~ outh,
                                          GRDTYPE %in% c("M", "E") ~ outm,
                                          GRDTYPE %in% c("S", "B") & is.na(outh) ~ outm))





cv_fun <- function(X, y, K=10){
  N <- nrow(X)
  folds <- cut(sample(N),breaks=K,labels=FALSE)
  
  cv.out <- foreach(k = 1:K, .combine=rbind) %do% {
    
    rf_mod <- randomForest(X[(folds != k),], y[(folds != k)],importance = TRUE)
    # delete imp = sort(rf$importance[,1]/sum(rf$importance[,1]),decreasing = T)[1:20]

    # delete rf_mod = randomForest(X[(folds != k),c(names(imp))], y[(folds != k)],importance = TRUE)

    rf_preds <- predict(rf_mod, newdata=X[(folds == k),])
    rf_mse <- mean((rf_preds-y[(folds == k)])^2)
    rf_r2 <- r2_calc(y[(folds == k)],rf_preds)
    
    
    it.ob <- c(rf_mse,rf_r2)
    names(it.ob) <- c("rf_mse","rf_r2")
    
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

y = data_m[,2]

data_m = data_m %>% select(starts_with('prem')) # 1394 * 5 for pre only

data_m <- data_m %>%
  mutate(across(everything(), ~ifelse(is.na(.x), 1, 0), .names="mis_{.col}"))

names(data_m) <- make.names(names(data_m))

ori_cov_name = names(data_m %>% select(-(starts_with('exist')|starts_with('GRDSPAN')|
                                starts_with('mis')|starts_with('out')|starts_with('CAMPUS'))))
data_m[ori_cov_name] = scale(data_m[ori_cov_name])

# replace na with 0
data_m[is.na(data_m)] <- 0 


#
## start testing ----

X = data_m


start_time <- Sys.time()
# #

#rf_mod <- randomForest(X, y,importance = TRUE)
#save(rf_mod, file = 'importance_m.Rdata')

middle_1 = cv_fun(X,y, K = 10)
#
end_time <- Sys.time()
end_time - start_time
#
save(middle_1,file = 'm_pre10_only.Rdata')
#
#load('middle-w/o-pre.Rdata')

# middle_zero
# cv_error = mean(middle_zero[,1])
# r2 = mean(middle_zero[,2])

# mean(middle_zero[,1])
# mean(middle_zero[,2])

# sort(rf_mod$importance[,1],decreasing = T)[1:50]





## high 0 ----
data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)
data_h = dummy_cols(data_h, select_columns = 'GRDSPAN')
data_h = subset(data_h,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY))

y = data_h[,2]

data_h = data_h %>% select(starts_with('preh')) # 1474 * 5 for pre only


data_h <- data_h %>%
  mutate(across(everything(), ~ifelse(is.na(.x), 1, 0), .names="mis_{.col}"))


names(data_h) <- make.names(names(data_h))

ori_cov_name = names(data_h %>% select(-(starts_with('exist')|starts_with('GRDSPAN')|
                                           starts_with('mis')|starts_with('out')|starts_with('CAMPUS'))))
data_h[ori_cov_name] = scale(data_h[ori_cov_name])



#data_h[,3:5145] = scale(data_h[,3:5145])
# replace na with 0
data_h[is.na(data_h)] <- 0 
#data_h[0,]

X = data_h


## start testing ----
start_time <- Sys.time()
#

high_1 = cv_fun(X,y, K = 10)

#rf_mod <- randomForest(X, y,importance = TRUE)
#save(rf_mod, file = 'importance_high.Rdata')

end_time <- Sys.time()
end_time - start_time

save(high_1,file = 'h_pre10_only.Rdata')

#load('high-w/o-pre.Rdata')
# 
# high_zero
# cv_error = mean(high_zero[,1])
# r2 = mean(high_zero[,2])
# 
# mean(high_zero[,1])
# mean(high_zero[,2])

# sort(rf_mod$importance[,1],decreasing = T)[1:50]

