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



cv_fun <- function(X, y,ntree,mtry, nodesize,maxnodes, K=10){
  N <- nrow(X)
  folds <- cut(sample(N),breaks=K,labels=FALSE)
  
  cv.out <- foreach(k = 1:K, .combine=rbind) %do% {
    
    rf_mod <- randomForest(X[(folds != k),], y[(folds != k)],importance = TRUE,ntree = ntree, mtry = mtry, 
                           nodesize = nodesize,maxnodes = maxnodes)
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
# everything above: for  w/ pre + all others
y = d[,2]

## for w/o pre + all others ----


# colnames.vec <- c()
# 
# for(i in 3:7){
#   yr <- paste0(i,i+1)
#   
#   ## ONLY non test
#   files <- list.files(paste0('../../data/raw/y',yr),pattern='.dat')
#   ## ONLY non test
#   filesupd <- files[!str_detect(files, "taks|ssi|cadcomp")]
#   
#   print(yr)
#   print(filesupd)
#   for(f in filesupd){
#     fname <- str_replace(f, "\\.dat", "")
#     print(fname)
#     
#     ### ONLY KEEP NAMES
#     suppressWarnings(suppressMessages(newdat <- read_csv(paste0('../../data/raw/y',yr,'/',f),col_names=FALSE,na=c('','.'))))
#     # if no column names, replace the dataset name in the front instead of "X"
#     names(newdat)[names(newdat)!='CAMPUS'] <- str_replace(names(newdat)[names(newdat)!='CAMPUS'],"X",paste0(fname, "_"))
#     
#     if(newdat[1,1]=='CAMPUS')
#       suppressMessages(newdat <- read_csv(paste0('../../data/raw/y',yr,'/',f),col_names=TRUE,na=c('.','')))
#     # add the year and dataset to the variable name
#     names(newdat)[names(newdat)!='CAMPUS'] <- paste0(names(newdat)[names(newdat)!='CAMPUS'],paste0("_", yr))
#     # set first name as "CAMPUS"
#     names(newdat)[1] <- 'CAMPUS'
#     print(newdat[1:2,1:6])
#     
#     colnames.vec <- c(colnames.vec,colnames(newdat))
#     
#     ### ONLY KEEP NAMES
#     #covs <- left_join(covs,newdat,'CAMPUS')
#     
#     
#     
#   }
# }

#d = d %>% select(contains(colnames.vec)) # 1394 * 2867



## for pre only + its indicator ---- 

# 
# colnames.vec <- c()
# 
# for(i in 3:7){
#   yr <- paste0(i,i+1)
#   
#   
#   files <- list.files(paste0('../../data/raw/y',yr),pattern='.dat')
#   ## ONLY test
#   filesupd <- files[str_detect(files, "taks|ssi|cadcomp")]
#   
#   print(yr)
#   print(filesupd)
#   for(f in filesupd){
#     fname <- str_replace(f, "\\.dat", "")
#     print(fname)
#     
#     ### ONLY KEEP NAMES
#     suppressWarnings(suppressMessages(newdat <- read_csv(paste0('../../data/raw/y',yr,'/',f),col_names=FALSE,na=c('','.'))))
#     # if no column names, replace the dataset name in the front instead of "X"
#     names(newdat)[names(newdat)!='CAMPUS'] <- str_replace(names(newdat)[names(newdat)!='CAMPUS'],"X",paste0(fname, "_"))
#     
#     if(newdat[1,1]=='CAMPUS')
#       suppressMessages(newdat <- read_csv(paste0('../../data/raw/y',yr,'/',f),col_names=TRUE,na=c('.','')))
#     # add the year and dataset to the variable name
#     names(newdat)[names(newdat)!='CAMPUS'] <- paste0(names(newdat)[names(newdat)!='CAMPUS'],paste0("_", yr))
#     # set first name as "CAMPUS"
#     names(newdat)[1] <- 'CAMPUS'
#     print(newdat[1:2,1:6])
#     
#     colnames.vec <- c(colnames.vec,colnames(newdat))
#     
#     ### ONLY KEEP NAMES
#     #covs <- left_join(covs,newdat,'CAMPUS')
#     
#     
#     
#   }
# }
# 
# d = d %>% select(contains(colnames.vec)) # 1394 * 3588



#
#
## start testing ----

X = d[,3:ncol(d)] # original
#y = d[,2]

#X = d[,2:ncol(d)] # no-pre or pre-only


start_time <- Sys.time()
# #


middle_1 = cv_fun(X,y,ntree = 500, mtry = 4000, nodesize = 20, maxnodes = 300, K = 10)
#
end_time <- Sys.time()
end_time - start_time
#
save(middle_1,file = 'm.Rdata')
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

# ## everything above: for  w/ pre + all others

# ## for w/o pre or pre-only ----

# d = d %>% select(contains(colnames.vec)) # 1474 * 2933(w/o) or 3874(pre-only)


# 
# ## define X and y ----
 X = d[,3:ncol(d)] # original
#X = d[,2:ncol(d)] # no-pre or pre-only
#y = d[,2]



## start testing ----
start_time <- Sys.time()
#

high_1 = cv_fun(X,y, ntree = 1000, mtry = 4000, nodesize = 10, maxnodes = 500, K = 10)


end_time <- Sys.time()
end_time - start_time

save(high_1,file = 'h.Rdata')

#load('high-w/o-pre.Rdata')
# 
# high_zero
# cv_error = mean(high_zero[,1])
# r2 = mean(high_zero[,2])
# 
# mean(high_zero[,1])
# mean(high_zero[,2])

# sort(rf_mod$importance[,1],decreasing = T)[1:50]

