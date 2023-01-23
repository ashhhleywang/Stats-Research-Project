library(tidyr)
library(dplyr)
library(missForest)
library(doParallel)
library(randomForest)
library('fastDummies')
library(neuralnet)
library(keras)
load("../../data/temp/remData.Rdata")

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



cv_fun <- function(X, y, K=10){
  N <- nrow(X)
  folds <- cut(sample(N),breaks=K,labels=FALSE)
  
  cv.out <- foreach(k = 1:K, .combine=rbind) %do% {
    
    rf <- randomForest(X[(folds != k),], y[(folds != k)],importance = TRUE)
    imp = sort(rf$importance[,1]/sum(rf$importance[,1]),decreasing = T)[1:20]
    
    rf_mod = randomForest(X[(folds != k),c(names(imp))], y[(folds != k)],importance = TRUE)
    
    rf_preds <- predict(rf_mod, newdata=X[(folds == k),c(names(imp))])
    rf_mse <- mean((rf_preds-y[(folds == k)])^2) 
    rf_r2 <- r2_calc(y[(folds == k)],rf_preds) 
    
    
    train_cov = X[(folds != k),c(names(imp))] 
    train_target = y[(folds != k)]
    test_cov = X[(folds == k),c(names(imp))]
    test_target = y[(folds == k)]
    
    covs <- array(data = as.matrix(train_cov), dim = c(nrow(train_cov),1,ncol(train_cov)))
    response <- array(data = as.matrix(train_target), dim = c(nrow(train_cov), 1))
    covs_test = array(data = as.matrix(test_cov), dim = c(nrow(test_cov),1,ncol(test_cov)))
    response_test = array(data = as.matrix(test_target), dim = c(nrow(test_cov), 1))
    
    
    input_layer <- layer_input(shape = c(1,ncol(train_cov)))
    output_layer <- input_layer %>%
      #layer_masking(mask_value = 0) %>%
      #layer_dropout(rate = .5) %>%
      #layer_lstm(units = 64, return_sequences = F, dropout = .5, recurrent_dropout = .5) %>%
      layer_dense(units = 8, activation = "sigmoid") %>% 
      #layer_dropout(rate = .5) %>%
      layer_dense(units = 1, activation = "linear")
    
    model <- keras_model(input_layer, output_layer)
    model %>% compile(
      loss = "mse",
      #metrics = "accuracy",
      optimizer = optimizer_adam()
    )
    
    
    history <- model %>% fit(
      x = covs,
      y = response,
      epochs = 200,
      validation_split = 0.25,
      verbose = 0,
      callbacks = list(print_dot_callback)
    )
    
    pred <- model %>% predict(x = covs_test)
    error_metric <- mean((pred[,1] - response_test)^2)
    r2_metric <- r2_calc(response_test,pred[,1])
    it.ob <- c(error_metric, r2_metric,rf_mse,rf_r2)
    names(it.ob) <- c("nn_mse","nn_r2","rf_mse","rf_r2")
    
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


#start from 3 becuz unlike b4, this time out does not have na
# for (i in 3:5145){
#   col = ifelse(is.na(data_m[i]), 1, 0)
#   data_m[paste0("col",i)] = col
#   #df$col2 <- ifelse(is.na(df[2]), 1, 0) # 1 = is NA
# }



# 
names(data_m) <- make.names(names(data_m))
data_m[,3:5145] = scale(data_m[,3:5145])
# replace na with 0
data_m[is.na(data_m)] <- 0 # 1394*10321

# data_m[1:5,1:5]
# data_m[1:5,5170:5180]
# data_m[1:5,9999:10000]
# duplicated_columns <- duplicated(as.list(select(data_m,starts_with('col'))))
# colnames(data_m[duplicated_columns])
# data_m_uniq = data_m[!duplicated_columns]

# data_m_uniq <- data_m[!duplicated(as.list(data_m))]
# 
# missing_cols[1:2,1:5]
# check[1:5]

missing_cols <- data_m %>%
  select(starts_with('mis'))
missing_colst <- t(missing_cols)
missing_colst <- data.frame(missing_colst) 

check <- missing_colst %>%
  distinct()

col_to_select = rownames(check)
mis_col= data_m %>% select(col_to_select)
d = cbind(data_m[,1:5178],mis_col)




# 
# 
# 
# 
# X = data_m[,3:ncol(data_m)]
# y = data_m[,2]
# 
# X[0:5,0:5]
# 
# 
# start_time <- Sys.time()
# #
# middle_zero = cv_fun(X,y,K = 10)
# 
# end_time <- Sys.time()
# end_time - start_time
# 
# save(middle_zero,file = 'cv_middle_0.Rdata')
# 
# load('cv_middle_0.Rdata')
# 
# middle_zero
# cv_error = mean(middle_zero[,1])
# r2 = mean(middle_zero[,2])

mean(middle_zero[,1])
mean(middle_zero[,2])



## middle mF ----
#list.files()
load("../fullcol.Rdata")
load("../no_na_middle.Rdata")
load("../no_na_high.Rdata")
middle_df = middle_result$ximp
data_m = check2 %>% filter((!is.na(outm))&(is.na(outh)))
data_m = inner_join(data_m, covsRem_noscale, by = 'CAMPUS')
data_m = data_m %>% filter(variable == "A08")
data_m = merge(data_m, grdXwalk)
library('fastDummies')
data_m = dummy_cols(data_m, select_columns = 'GRDSPAN') # 41 levels under GRDSPAN without na in outm
data_m = subset(data_m,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY)) # dont need to change out to outm becuz

for (i in 3:5145){
  col = ifelse(is.na(data_m[i]), 1, 0)
  data_m[paste0("col",i)] = col
  #df$col2 <- ifelse(is.na(df[2]), 1, 0) # 1 = is NA 
}
#data_m[,5146:10316]# 5146 - 10316 are indicator col

middle_df = cbind(data_m[,5146:10316],middle_df)
#middle_df[0,]
#middle_df[,5172] # 1 - 5171 col are indicator, where the first 28 are GRDSPAN

middle_df = cbind(data_m$CAMPUS,middle_df)
middle_df = cbind(data_m$out,middle_df) # 1394 * 10269

names(middle_df)[names(middle_df) == 'data_m$CAMPUS'] <- 'CAMPUS'
names(middle_df)[names(middle_df) == 'data_m$out'] <- 'out'
#middle_df = left_join(middle_df, grdXwalk, by = 'CAMPUS') # 1394 * 5101 (5098 + 3 in grdX)
middle_df[0,5173:5174] 

#middle_df  = dummy_cols(middle_df, select_columns = 'GRDSPAN') # 28 indicator col
#middle_df = subset(middle_df,select = -c(GRDTYPE,GRDSPAN,COUNTY)) # 1394 * 5126

middle_df$exist34 = ifelse(middle_df$CAMPUS %in% name_34,1,0)
middle_df$exist45 = ifelse(middle_df$CAMPUS %in% name_45,1,0)
middle_df$exist56 = ifelse(middle_df$CAMPUS %in% name_56,1,0)
middle_df$exist67 = ifelse(middle_df$CAMPUS %in% name_67,1,0)
middle_df$exist78 = ifelse(middle_df$CAMPUS %in% name_78,1,0) # 1394 * 10274


#middle_df[0,5174:10269] col 5174:10269 are cov cols to scale

middle_df[,5174:10269] = scale(middle_df[,5174:10269])
# replace na with 0
middle_df[is.na(middle_df)] <- 0
names(middle_df) <- make.names(names(middle_df))

# middle_df <- as.matrix(middle_df)
# dimnames(middle_df) <- NULL
# middle_df = matrix(as.numeric(middle_df),ncol = ncol(middle_df)) # 1394*10274



X =middle_df[,3:ncol(middle_df)] # 1394 * 10272
y = middle_df[,1] # 1394 * 1



start_time <- Sys.time()
# 
middle_mF= cv_fun(X,y,K = 10)

end_time <- Sys.time()
end_time - start_time

save(middle_mF,file = 'cv_middle_mF.Rdata')

load('cv_middle_mF.Rdata')

middle_mF
cv_error = mean(middle_mF[,1])
r2 = mean(middle_mF[,2])

mean(middle_mF[,1])
mean(middle_mF[,2])


## high 0 ----
data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)
data_h = dummy_cols(data_h, select_columns = 'GRDSPAN') 
data_h = subset(data_h,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY)) 
#names(which(colSums(is.na(data_h))>0))

#start from 3 becuz unlike b4, this time out does not have na
for (i in 3:5145){
  col = ifelse(is.na(data_h[i]), 1, 0)
  data_h[paste0("col",i)] = col
  #df$col2 <- ifelse(is.na(df[2]), 1, 0) # 1 = is NA 
}

data_h$exist34 = ifelse(data_h$CAMPUS %in% name_34,1,0)
data_h$exist45 = ifelse(data_h$CAMPUS %in% name_45,1,0)
data_h$exist56 = ifelse(data_h$CAMPUS %in% name_56,1,0)
data_h$exist67 = ifelse(data_h$CAMPUS %in% name_67,1,0)
data_h$exist78 = ifelse(data_h$CAMPUS %in% name_78,1,0)
#names(which(colSums(is.na(data_h))>0)) 

names(data_h) <- make.names(names(data_h))
data_h[,3:5145] = scale(data_h[,3:5145])
# replace na with 0
data_h[is.na(data_h)] <- 0 # 1474* 10322
#data_h[0,]


X = data_h[,3:ncol(data_h)] # 1474 * 10320
y = data_h[,2] # 1474 * 1

start_time <- Sys.time()
#
high_zero = cv_fun(X,y,K = 10)

end_time <- Sys.time()
end_time - start_time

save(high_zero,file = 'cv_high_0.Rdata')

load('cv_high_0.Rdata')

high_zero
cv_error = mean(high_zero[,1])
r2 = mean(high_zero[,2])

mean(high_zero[,1])
mean(high_zero[,2])

## high_mF ----
high_df = high_result$ximp # 1474 * 5143 no additional col added

# getting the campus and out columns back
data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)

data_h = dummy_cols(data_h, select_columns = 'GRDSPAN') 
data_h = subset(data_h,select = -c(GRDTYPE,Type,variable,outm,outh,GRDSPAN,COUNTY)) 
for (i in 3:5145){
  col = ifelse(is.na(data_h[i]), 1, 0)
  data_h[paste0("col",i)] = col
  #df$col2 <- ifelse(is.na(df[2]), 1, 0) # 1 = is NA 
}
data_h[,5146] # 1474*10317

high_df = cbind(data_h[,5146:10317],high_df) #1474*10315

high_df = cbind(data_h$CAMPUS,high_df)
high_df = cbind(data_h$out,high_df)

names(high_df)[names(high_df) == 'data_h$CAMPUS'] <- 'CAMPUS'
names(high_df)[names(high_df) == 'data_h$out'] <- 'out'

# 1474 * 10317

high_df$exist34 = ifelse(high_df$CAMPUS %in% name_34,1,0)
high_df$exist45 = ifelse(high_df$CAMPUS %in% name_45,1,0)
high_df$exist56 = ifelse(high_df$CAMPUS %in% name_56,1,0)
high_df$exist67 = ifelse(high_df$CAMPUS %in% name_67,1,0)
high_df$exist78 = ifelse(high_df$CAMPUS %in% name_78,1,0) # 1474 * 10322

high_df[0,10317:10322] # 5175 - 10317 are cov col to scale

# scale
high_df[,5175:10317] = scale(high_df[,5175:10317]) #scaling returns na for zero variance columns: meaning that they are all the same value

# replace na with 0
high_df[is.na(high_df)] <- 0
names(high_df) <- make.names(names(high_df))

# high_df <- as.matrix(high_df)
# dimnames(high_df) <- NULL
# high_df = matrix(as.numeric(high_df),ncol = ncol(high_df)) # 1474*10322



X =high_df[,3:ncol(high_df)] # 1474 * 10320
y = high_df[,1] # 1474 * 1




start_time <- Sys.time()
# 
high_mF= cv_fun(X,y,K = 10)

end_time <- Sys.time()
end_time - start_time

save(high_mF,file = 'cv_high_mF.Rdata')

load('cv_high_mF.Rdata')

high_mF
cv_error = mean(high_mF[,1])
r2 = mean(high_mF[,2])

mean(high_mF[,1])
mean(high_mF[,2])

