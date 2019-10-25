# Statistics module

## Statistical hypothesis tests

### Definition
A statistical hypothesis test is a method of statistical inference. 
Commonly, two statistical data sets are compared. 
_A hypothesis is proposed for the statistical relationship between the two 
data sets, and this is compared as an alternative to an idealized null 
hypothesis that proposes no relationship between two data sets. 
**The comparison is deemed statistically significant if the relationship 
between the data sets would be an unlikely realization of the null hypothesis 
according to a threshold probability - the significance level.**_ 

Hypothesis tests are used when determining what outcomes of a study would 
lead to a rejection of the null hypothesis for a pre-specified 
level of significance.
The process of distinguishing between the null hypothesis and the alternative 
hypothesis is aided by considering two conceptual types of errors. 
The first type of error occurs when the null hypothesis is wrongly rejected. 
The second type of error occurs when the null hypothesis is wrongly not rejected. 
(The two types are known as type 1 and type 2 errors.)

Hypothesis tests based on statistical significance are another way of expressing 
confidence intervals (more precisely, confidence sets). 
In other words, every hypothesis test based on significance can be obtained 
via a confidence interval, and every confidence interval can be obtained via 
a hypothesis test based on significance.

### Normality tests

#### Shapiro-Wilk Test
Assumptions:
- observations in each sample are independent and identically distributed (iid). 
    
Interpretation: 
- H0: the sample has a Gaussian distribution.
- H1: the sample does not have a Gaussian distribution.
                    
#### D'Agostino's K^2 Test
Assumptions: 
- observations in each sample are independent and identically distributed (iid).

Interpretation: 
- H0: the sample has a Gaussian distribution.
- H1: the sample does not have a Gaussian distribution.
                    
#### Anderson-Darling Test
Assumptions: 
- observations in each sample are independent and identically distributed (iid).

Interpretation: 
- H0: the sample has a Gaussian distribution.
- H1: the sample does not have a Gaussian distribution.
                    
### Correlation Tests

#### Pearson's Correlation Coefficient
Tests whether two samples have a linear relationship.
    
Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample are normally distributed.
- observations in each sample have the same variance.

Interpretation: 
- H0: the two samples are independent.
- H1: there is a dependency between the samples.

#### Spearman's Rank Correlation
Tests whether two samples have a monotonic relationship.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.

Interpretation: 
- H0: the two samples are independent.
- H1: there is a dependency between the samples.

#### Kendall's Rank Correlation
Tests whether two samples have a monotonic relationship.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.

Interpretation: 
- H0: the two samples are independent.
- H1: there is a dependency between the samples.
                    
#### Chi-Squared Test
Tests whether two categorical variables are related or independent.

Assumptions: 
- observations used in the calculation of the contingency table are independent.
- 25 or more examples in each cell of the contingency table.
    
Interpretation: 
- H0: the two samples are independent.
- H1: there is a dependency between the samples.
                    
### Parametric Statistical Hypothesis Tests

#### Student's t-test
Tests whether the means of two independent samples are significantly different.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample are normally distributed.
- observations in each sample have the same variance.

Interpretation: 
- H0: the means of the samples are equal.
- H1: the means of the samples are unequal.

#### Paired Student’s t-test
Tests whether the means of two paired samples are significantly different.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample are normally distributed.
- observations in each sample have the same variance.
- observations across each sample are paired.
    
Interpretation: 
- H0: the means of the samples are equal.
- H1: the means of the samples are unequal.

#### Analysis of Variance Test (ANOVA)
Tests whether the means of two or more independent
samples are significantly different.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample are normally distributed.
- observations in each sample have the same variance.

Interpretation: 
- H0: the means of the samples are equal.
- H1: one or more of the means of the samples are unequal.
                    
### Nonparametric Statistical Hypothesis Tests

#### Mann-Whitney U Test
Tests whether the distributions of two independent samples are equal or not.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.

Interpretation: 
- H0: the distributions of both samples are equal.
- H1: the distributions of both samples are not equal.

#### Wilcoxon Signed-Rank Test
Tests whether the distributions of two paired samples are equal or not.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.
- observations across each sample are paired.

Interpretation: 
- H0: the distributions of both samples are equal.
- H1: the distributions of both samples are not equal.

#### Kruskal-Wallis H Test
Tests whether the distributions of two or more independent samples are equal or not.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.

Interpretation: 
- H0: the distributions of all samples are equal.
- H1: the distributions of one or more samples are not equal.

#### Friedman Test
Tests whether the distributions of two or more paired samples are equal or not.

Assumptions: 
- observations in each sample are independent and identically distributed (iid).
- observations in each sample can be ranked.
- observations across each sample are paired.

Interpretation: 
- H0: the distributions of all samples are equal.
- H1: the distributions of one or more samples are not equal.



## Confidence intervals

### Definition

In statistics, **_a confidence interval (CI) is a type of interval estimate, 
computed from the statistics of the observed data, that might contain the 
true value of an unknown population parameter._** 
The interval has an associated confidence level, or coverage that, 
loosely speaking, quantifies the level of confidence that the 
deterministic parameter is captured by the interval. 
More strictly speaking, **_the confidence level represents 
the frequency (i.e. the proportion) of possible confidence intervals 
that contain the true value of the unknown population parameter._** 
In other words, if confidence intervals are constructed using 
a given confidence level from an infinite number of independent 
sample statistics, the proportion of those intervals that contain 
the true value of the parameter will be equal to the confidence level.

Confidence intervals consist of a range of potential values of the 
unknown population parameter. 
However, **the interval computed from a particular sample does 
not necessarily include the true value of the parameter.** 
Based on the (usually taken) assumption that observed data are 
random samples from a true population, **the confidence interval 
obtained from the data is also random.**

The confidence level is designated prior to examining the data. 
Most commonly, the 95% confidence level is used. 
However, other confidence levels can be used, for example, 90% and 99%.

**Factors affecting the width of the confidence interval include 
the size of the sample, the confidence level, and the variability in the sample.** 
A larger sample will tend to produce a better estimate of 
the population parameter, when all other factors are equal. 
A higher confidence level will tend to produce a broader confidence interval.
