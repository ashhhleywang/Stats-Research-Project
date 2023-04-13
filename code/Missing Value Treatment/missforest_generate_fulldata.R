library(tidyr)
library(dplyr)
library(missForest)
library(doParallel)



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

check_na = check2[is.na(check2$out),]
Counting = check_na %>% count(CAMPUS)


#names(covsRem_noscale)

## ---- 
# run through all columns
# run_all = covsRem_noscale[,2:(ncol(covsRem_noscale)-1)]
# no_cores <- detectCores() - 1
# cl <- makeCluster(no_cores)
# registerDoParallel(cl)
# 
# start_time <- Sys.time()
# run_all_result = missForest(run_all,parallelize = "variables", verbose = T)
# stopCluster(cl)
# 
# end_time <- Sys.time()
# end_time - start_time
# save(run_all_result,file = 'fullcol.Rdata')
# load('fullcol.Rdata')
# run_all_result$ximp
# all variables (excluding first and last column since not numerical): ? iteration, total time = 22hr

##----
# replace na in middle data set
# 1394 middle schools

data_m = check2 %>% filter((!is.na(outm))&(is.na(outh)))
#double check
names(which(colSums(is.na(data_m))==0))

data_m = inner_join(data_m, covsRem_noscale, by = 'CAMPUS')
data_m = data_m %>% filter(variable == "A08")
data_m = merge(data_m, grdXwalk)



middle_na = covsRem_noscale[covsRem_noscale$CAMPUS %in% data_m$CAMPUS,2:(ncol(covsRem_noscale)-1)]

sum(is.na(middle_na$ccadcomp3_133_34)) # ccadcomp3_133_34 is full of na -> after fillig na using missForest, this col is deleted
# thus missforest automatically remove columns all with na

# 
# no_cores <- detectCores() - 1
# cl <- makeCluster(no_cores)
# registerDoParallel(cl)
# 
# start_time <- Sys.time()
# middle_result = missForest(middle_na,parallelize = "variables", verbose = T)
# stopCluster(cl)
# 
# end_time <- Sys.time()
# end_time - start_time
# save(middle_result,file = 'no_na_middle.Rdata')
# load('no_na_middle.Rdata')
# middle_result







## ----
# replace na in high data set 
# 1474 high schools

data_h = check2 %>% filter(!is.na(outh))
data_h = inner_join(data_h, covsRem_noscale, by = 'CAMPUS')
data_h = data_h %>% filter(variable == "A08")
data_h = merge(data_h, grdXwalk)
high_na = covsRem_noscale[covsRem_noscale$CAMPUS %in% data_h$CAMPUS,2:(ncol(covsRem_noscale)-1)]


no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
registerDoParallel(cl)

start_time <- Sys.time()
high_result = missForest(high_na,parallelize = "variables", verbose = T)
stopCluster(cl)

end_time <- Sys.time()
end_time - start_time
save(high_result,file = 'no_na_high.Rdata')
load('no_na_high.Rdata')
high_result



