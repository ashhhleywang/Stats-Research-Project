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

library(foreach)

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




cv_fun <- function(X, y, maxnodes_range, K = 10,seed = 1) {
  
  set.seed(seed)
  N <- nrow(X)
  folds <- cut(sample(N), breaks = K, labels = FALSE)
  
  cv_results <- list()
  for (maxnodes in maxnodes_range) {
    cv.out <- foreach(k = 1:K, .combine = rbind) %dopar% {
      
      start_time <- Sys.time()
      
      rf_mod <- randomForest(X[(folds != k),], y[(folds != k)],importance = TRUE, maxnodes = maxnodes)
      rf_preds <- predict(rf_mod, newdata = X[(folds == k),])
      rf_mse <- mean((rf_preds - y[(folds == k)])^2)
      rf_r2 <- r2_calc(y[(folds == k)], rf_preds)
      
      end_time <- Sys.time()
      
      it.ob <- c(rf_mse, rf_r2,time_taken = as.numeric(end_time - start_time))
      names(it.ob) <- c("rf_mse", "rf_r2","time_taken")
      
      it.ob
    }
    
    cv_results[[as.character(maxnodes)]] <- cv.out
  }
  
  return(cv_results)
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

# data_m = data_m %>% select(-starts_with('pre')) # 1394 * 5168 for w/o pre + all others

data_m <- data_m %>%
  mutate(across(everything(), ~ifelse(is.na(.x), 1, 0), .names="mis_{.col}"))

names(data_m) <- make.names(names(data_m))

ori_cov_name = names(data_m %>% select(-(starts_with('exist')|starts_with('GRDSPAN')|
                                starts_with('mis')|starts_with('out')|starts_with('CAMPUS'))))
data_m[ori_cov_name] = scale(data_m[ori_cov_name])

# replace na with 0
data_m[is.na(data_m)] <- 0 # 1394*10356

missing_cols <- data_m %>%
  select(starts_with('mis'))
missing_colst <- t(missing_cols)
missing_colst <- data.frame(missing_colst)

check <- missing_colst %>%
  distinct()

col_to_select = rownames(check)
mis_col= data_m %>% select(col_to_select)

everything_not_mis = names(data_m %>% select(-starts_with('mis')))
#data_m[everything_not_mis]
d = cbind(data_m[everything_not_mis],mis_col) # 5146 - 5173 are GRDSPAN indicator, 5174-5178 are exist34/45/56...
# 1394* 6502

y = d[,2]

## start testing ----

X = d[,3:ncol(d)] # original


# X = d[1:200,3:20] 
# y = d[1:200,2]

start_time <- Sys.time()
# #



maxnodes_range = c(150,200,300,400,500,600,1000)

numCores <- detectCores()
registerDoParallel(numCores)
middle_1 = cv_fun(X,y,maxnodes_range, K = 10,seed = 1)
stopImplicitCluster()

#
end_time <- Sys.time()
end_time - start_time
#


Matrix_m <- matrix(unlist(lapply(middle_1, function(x) colMeans(x))), ncol = 3, byrow = T)
colnames(Matrix_m) = c('rf_mse','rf_r2','time_taken')
rownames(Matrix_m) = c(as.numeric(names(middle_1)))

save(Matrix_m,file = 'm_maxnodes.Rdata')


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

ori_cov_name = names(data_h %>% select(-(starts_with('exist')|starts_with('GRDSPAN')|
                                           starts_with('mis')|starts_with('out')|starts_with('CAMPUS'))))
data_h[ori_cov_name] = scale(data_h[ori_cov_name])



#data_h[,3:5145] = scale(data_h[,3:5145])
# replace na with 0
data_h[is.na(data_h)] <- 0 # 1474* 10358
#data_h[0,]

missing_cols <- data_h %>%
  select(starts_with('mis'))
missing_colst <- t(missing_cols)
missing_colst <- data.frame(missing_colst)

check <- missing_colst %>%
  distinct()

col_to_select = rownames(check)
mis_col= data_h %>% select(col_to_select)

everything_not_mis = names(data_h %>% select(-starts_with('mis')))
#data_h[everything_not_mis]
d = cbind(data_h[everything_not_mis],mis_col) #1474 * 6859
y = d[,2]


# 
# ## define X and y ----
X = d[,3:ncol(d)] # original


## start testing ----
start_time <- Sys.time()
#

numCores <- detectCores()
registerDoParallel(numCores)
high_1 = cv_fun(X,y, maxnodes_range, K = 10,seed = 1)
stopImplicitCluster()

end_time <- Sys.time()
end_time - start_time




#as.numeric(names(high_1))

#lapply(high_1, function(x) colMeans(x))


Matrix_h <- matrix(unlist(lapply(high_1, function(x) colMeans(x))), ncol = 3, byrow = T)
colnames(Matrix_h) = c('rf_mse','rf_r2','time_taken')
rownames(Matrix_h) = c(as.numeric(names(high_1)))
save(Matrix_h,file = 'h_maxnodes.Rdata')

plot(as.numeric(rownames(Matrix_m)),Matrix_m[,1],xlab="maxnodes", ylab="MSE",type = 'b',main = 'middle')
plot(as.numeric(rownames(Matrix_h)),Matrix_h[,1],xlab="maxnodes", ylab="MSE",type = 'b',main = 'high')

