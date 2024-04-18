# https://mp.weixin.qq.com/s/6UG3x7_AvoXT3urQfOtFAg

library(CBCgrps)

head(iris)

table(iris$Species)

dat_3species <- iris
dat_2species <- iris[iris$Species %in% names(table(iris$Species))[1:2],]
dat_2species$Species <- as.character(dat_2species$Species)
table(dat_2species$Species)

# 分组变量，它有2个组，分别分析其他变量在这2组中有没有差异。
# 因为这个数据中既有分类变量（比如gender），也有连续型变量，这两种变量使用的统计方法是完全不一样的，
# 一般来说计数资料是卡方检验，而计量资料使用的是t检验或者秩和检验。

tab2 <- twogrps(dat_2species, gvar = "Species", ShowStatistic = T)
tab2

# 对于符合正态分布的计量资料，这里用的是t检验，如果不符合使用的是wilcoxon秩和检验，对于计数资料则使用的是卡方检验。

# 多变量：
# 符合正态分布的会使用方差分析，不符合的会使用Kruskal-Wallis检验，分类变量会使用卡方检验或者Fisher精确概率法

tab3 <- multigrps(dat_3species, gvar = "Species", ShowStatistic = T)
tab3

# 拓展dlookr 正态性检验可视化
library(dlookr)
plot_normality(iris)

