# https://mp.weixin.qq.com/s/EuP2LTqJbbWOy4s_4QphHA
# https://dzhakparov.github.io/GeneSelectR/vignettes/example.html

rm(list=ls())
library(tidyverse)
library(GeneSelectR)

GeneSelectR::set_reticulate_python()
GeneSelectR::configure_environment()

load("./01_UrbanRandomSubset.rda")
DT::datatable(UrbanRandomSubset)
table(UrbanRandomSubset$treatment)

X <- UrbanRandomSubset %>% select(-treatment)
y <- UrbanRandomSubset$treatment %>% as.factor() %>% as.integer()


selection_results <- GeneSelectR(X = X, 
                                 y = y,
                                 njobs = 48)
selection_results


plot_feature_importance(selection_results, top_n_features = 5)
plot_metrics(selection_results)


selection_results@test_metrics 
selection_results@cv_mean_score


overlap <- calculate_overlap_coefficients(selection_results)
overlap

plot_overlap_heatmaps(overlap)

custom_list <- list(custom_list = c('char1','char2','char3','char4','char5'),
                    custom_list2 = c('char1','char2','char3','char4','char5'))
overlap1 <- calculate_overlap_coefficients(selection_results, custom_lists = custom_list)
plot_overlap_heatmaps(overlap1)
plot_upset(selection_results, custom_lists = custom_list)
