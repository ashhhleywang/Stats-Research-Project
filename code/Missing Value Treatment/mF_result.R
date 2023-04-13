#----
load('fullcol.RData')
load('no_na_middle.RData')
load('no_na_high.RData')

library(tidyr)
library(dplyr)
library(missForest)
library(randomForest)
library(doParallel)
library("Metrics")
library('fastDummies')
r2_calc <- function(classifications, predictions){
  
  idx <- !is.na(predictions) & !is.na(classifications)
  classifications <- classifications[idx]
  predictions <- predictions[idx]
  
  1 - (sum((classifications-predictions)^2)/sum((classifications-mean(classifications))^2))
  
}

load("../data/temp/remData.Rdata")

#Delete columns end with _na
df = taksRem %>% select(-contains("_na"))

df2 = inner_join(df, grdXwalk, by = 'CAMPUS')
df2 = subset(df2,select = -c(COUNTY,GRDSPAN))
check = df2 %>% pivot_longer(cols = -c(CAMPUS,GRDTYPE), names_to = c("type","variable"),names_pattern = "(.+)([A-Z]\\d{2})",values_to = "value")

check = check %>% pivot_wider(names_from = type,values_from = value)

check2 = check %>% mutate(out = case_when(GRDTYPE %in% c("S", "B") & !is.na(outh) ~ outh,
                                          GRDTYPE %in% c("M", "E") ~ outm,
                                          GRDTYPE %in% c("S", "B") & is.na(outh) ~ outm))


#check2[check2$CAMPUS == 101912218,]

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



#run_all_result$ximp[0:2]
dim(run_all_result$ximp) #  2985 * 5143 (cuz excluding CAMPUS and TYPE)

dim(middle_result$ximp) # 1394 * 5096 has 5096 col becuz there are col with not a single value (all na)

dim(high_result$ximp) #  1474 * 5143

for (i in colnames(run_all_result$ximp)){
  if (!(i %in% colnames(middle_result$ximp))){
    print(i)
  }
}
 

#----
# create GRDTYPE indicator col by joining GRDSPAN
middle_df = middle_result$ximp # 1394 * 5096 no additional col added




#middle_df$CPETBLAP_78

# getting the campus and out columns back
data_m = check2 %>% filter((!is.na(outm))&(is.na(outh)))
data_m = inner_join(data_m, covsRem_noscale, by = 'CAMPUS')
data_m = data_m %>% filter(variable == "A08")
data_m = merge(data_m, grdXwalk)

middle_df = cbind(data_m$CAMPUS,middle_df)
middle_df = cbind(data_m$out,middle_df)

names(middle_df)[names(middle_df) == 'data_m$CAMPUS'] <- 'CAMPUS'
names(middle_df)[names(middle_df) == 'data_m$out'] <- 'out'


# 1394 * 5098
names(which(colSums(is.na(middle_df))>0))  # no col has na

middle_df = left_join(middle_df, grdXwalk, by = 'CAMPUS') # 1394 * 5101 (5098 + 3 in grdX)


middle_df  = dummy_cols(middle_df, select_columns = 'GRDSPAN') # 28 indicator col
middle_df = subset(middle_df,select = -c(GRDTYPE,GRDSPAN,COUNTY)) # 1394 * 5126

middle_df$exist34 = ifelse(middle_df$CAMPUS %in% name_34,1,0)
middle_df$exist45 = ifelse(middle_df$CAMPUS %in% name_45,1,0)
middle_df$exist56 = ifelse(middle_df$CAMPUS %in% name_56,1,0)
middle_df$exist67 = ifelse(middle_df$CAMPUS %in% name_67,1,0)
middle_df$exist78 = ifelse(middle_df$CAMPUS %in% name_78,1,0) # 1394 * 5131



names(which(colSums(is.na(middle_df))>0)) 



# scale
middle_df[,3:5098] = scale(middle_df[,3:5098]) #scaling returns na for zero variance columns: meaning that they are all the same value

#names(which(colSums(is.na(middle_df))>0)) 

# replace na with 0
middle_df[is.na(middle_df)] <- 0
names(which(colSums(is.na(middle_df))>0)) 
names(middle_df) <- make.names(names(middle_df))


# train
set.seed(2)
n_rows<- nrow(middle_df)
n_cols <- ncol(middle_df)
train_idx <- sample(seq(n_rows), size=floor(0.7*n_rows))
train_mid <- middle_df[train_idx,]
test_mid <- middle_df[-train_idx,]
var(middle_df$out) #171.1166
var(train_mid$out) # 173.9381
var(test_mid$out) # 164.9138


start_time <- Sys.time()
rf_mod2 <- randomForest(
  out~. - CAMPUS, # Model formula
  data=train_mid, # Training data mtry=n_cols_boston-1, # Use all columns
  importance=TRUE)
end_time <- Sys.time()
end_time - start_time
# Time difference of 6.648685 mins

rf_mod2
# Call:
#   randomForest(formula = out ~ . - CAMPUS, data = train_mid, importance = TRUE) 
# Type of random forest: regression
# Number of trees: 500
# No. of variables tried at each split: 1709
# 
# Mean of squared residuals: 57.93029
# % Var explained: 66.66

oob_preds = predict(rf_mod2)
oob_mse <- mean((oob_preds-train_mid$out)^2) #   57.93029
oob_r2 <- r2_calc(train_mid$out,oob_preds) #  0.6666069

test_preds <- predict(rf_mod2, newdata=test_mid)
test_mse <- mean((test_preds-test_mid$out)^2) #  46.44482
test_r2 <- r2_calc(test_mid$out,test_preds) # 0.7176953

save(rf_mod2,file = 'rf_middle_nonan.Rdata')
load('rf_middle_nonan.Rdata')


library(gbm)
start_time <- Sys.time()
boost_mod <- gbm(
  out~.-CAMPUS,
  data=train_mid,
  distribution="gaussian")
  #n.trees=5000,
  #interaction.depth=4,
  #shrinkage=0.01)
end_time <- Sys.time()
end_time - start_time

# Time difference of 18.54487 secs

boost_mod
# A gradient boosted model with gaussian loss function.
# 5000 iterations were performed.
# There were 5129 predictors of which 3841 had non-zero influence.

oob_preds = predict(boost_mod)
oob_mse <- mean((oob_preds-train_mid$out)^2) #  37.26327
oob_r2 <- r2_calc(train_mid$out,oob_preds) # 0.7855471

test_preds <- predict(boost_mod, newdata=test_mid)
test_mse <- mean((test_preds-test_mid$out)^2) #   50.45345
test_r2 <- r2_calc(test_mid$out,test_preds) # 0.6933297

save(boost_mod,file = 'gb_middle_nonan.Rdata')
load('gb_middle_nonan.Rdata')



#---- 
# high school

# create GRDTYPE indicator col by joining GRDSPAN
high_df = high_result$ximp # 1474 * 5143 no additional col added

# getting the campus and out columns back
data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)

high_df = cbind(data_h$CAMPUS,high_df)
high_df = cbind(data_h$out,high_df)

names(high_df)[names(high_df) == 'data_h$CAMPUS'] <- 'CAMPUS'
names(high_df)[names(high_df) == 'data_h$out'] <- 'out'

# 1474 * 5145
names(which(colSums(is.na(high_df))>0)) 

high_df = left_join(high_df, grdXwalk, by = 'CAMPUS') # 1474 * 5148 

high_df  = dummy_cols(high_df, select_columns = 'GRDSPAN') # 29 indicator col
high_df = subset(high_df,select = -c(GRDTYPE,GRDSPAN,COUNTY)) # 1474 * 5174

high_df$exist34 = ifelse(high_df$CAMPUS %in% name_34,1,0)
high_df$exist45 = ifelse(high_df$CAMPUS %in% name_45,1,0)
high_df$exist56 = ifelse(high_df$CAMPUS %in% name_56,1,0)
high_df$exist67 = ifelse(high_df$CAMPUS %in% name_67,1,0)
high_df$exist78 = ifelse(high_df$CAMPUS %in% name_78,1,0) # 1474 * 5179

names(which(colSums(is.na(high_df))>0)) 


# scale
high_df[,3:5145] = scale(high_df[,3:5145]) #scaling returns na for zero variance columns: meaning that they are all the same value

names(which(colSums(is.na(high_df))>0)) 

# replace na with 0
high_df[is.na(high_df)] <- 0
names(which(colSums(is.na(high_df))>0)) 
names(high_df) <- make.names(names(high_df))


# train
set.seed(2)
n_rows<- nrow(high_df)
n_cols <- ncol(high_df)


train_idx <- sample(seq(n_rows), size=floor(0.7*n_rows))
train_mid <- high_df[train_idx,]
test_mid <- high_df[-train_idx,]
var(high_df$out) # 455.9676
var(train_mid$out) # 446.3356
var(test_mid$out) # 479.4284

start_time <- Sys.time()
rf_mod3 <- randomForest(
  out~. - CAMPUS, # Model formula
  data=train_mid, # Training data mtry=n_cols_boston-1, # Use all columns
  importance=TRUE)
end_time <- Sys.time()
end_time - start_time
# Time difference of 7.583455 mins

rf_mod3

# Call:
#   randomForest(formula = out ~ . - CAMPUS, data = train_mid, importance = TRUE) 
# Type of random forest: regression
# Number of trees: 500
# No. of variables tried at each split: 1725
# 
# Mean of squared residuals: 126.1203
# % Var explained: 71.72


test_preds <- predict(rf_mod3, newdata=test_mid)
test_mse <- mean((test_preds-test_mid$out)^2) #  139.3613
test_r2 <- r2_calc(test_mid$out,test_preds) #  0.7086602

save(rf_mod3,file = 'rf_high_nonan.Rdata')
load('rf_high_nonan.Rdata')


start_time <- Sys.time()
boost_mod <- gbm(
  out~.-CAMPUS,
  data=train_mid,
  distribution="gaussian")
#n.trees=5000,
#interaction.depth=4,
#shrinkage=0.01)
end_time <- Sys.time()
end_time - start_time

# Time difference of 18.54487 secs

boost_mod
# A gradient boosted model with gaussian loss function.
# 5000 iterations were performed.
# There were 5129 predictors of which 3841 had non-zero influence.

oob_preds = predict(boost_mod)
oob_mse <- mean((oob_preds-train_mid$out)^2) #  37.26327
oob_r2 <- r2_calc(train_mid$out,oob_preds) # 0.7855471

test_preds <- predict(boost_mod, newdata=test_mid)
test_mse <- mean((test_preds-test_mid$out)^2) #   50.45345
test_r2 <- r2_calc(test_mid$out,test_preds) # 0.6933297

save(boost_mod,file = 'gb_high_nonan.Rdata')
load('gb_high_nonan.Rdata')


