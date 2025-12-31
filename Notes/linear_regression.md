# Linear Regression
### 1. Estimating Coefficients
$$
\begin{align*}
\LARGE Y= \beta_0 + \beta_1 X + \epsilon
\end{align*}
$$
Here $\beta_0$ is the intercept term- that is, the expected value of $Y$ when $X=0$, and $\beta_1$ is the slope- the average increase in $Y$ associated with a one unit increase in $X$. The error term is a catch-all for what we miss with this simple model; the true relationship is probably not linear, there may be other variables that cause variation in $Y$, and there may be measurement error. **We typically assume that the error term is independent of $X$**.The above equation is called population regression equation. 

The unknown coefficients $\beta_0$ and $\beta_1$ in the popluation regression line is unknown. We estimate these unknown coefficients using the principle of least sqaures. 

$$
\begin{align*}
\LARGE \hat{\beta_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}
\end{align*}
$$

$$
\begin{align*}
\LARGE \hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}
\end{align*}
$$

###  2. Standard Errors of Estimates 
The least squares coefficient estimates are unbiased. This means that if we estimate $\beta_0$ and $\beta_1$ on the basis of a particular data set, then our estimates won't be equal to $\beta_0$ and $\beta_1$. But if we could average the estimates obtained would be spot on. 

How accurate are the estimates? We answer this question by computing the standard errors of the estimates. 

$$
\begin{align*}
\LARGE SE(\hat{\beta_0})^2 = \sigma^2[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n}(x_i-\bar{x})^2}]
\end{align*}
$$

$$
\begin{align*}
\LARGE SE(\hat{\beta_0})^2 = \frac{\sigma^2}{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\end{align*}
$$

where $\sigma^2 = Var(\epsilon)$

These formulas to be strictly valid, we need to assume that the errors for each observation are uncorrelated with common variance $\sigma^2$. We see that $SE(\hat{\beta_0})$ would be the same as $SE(\hat{\mu})$ if $\bar{x}$ were zero.$SE(\hat{\beta_1})$ is smaller when $x_i$ are more spread out; intuitively we have more leverage to estimate a slope when this is the case. In general, $\sigma^2$ is not known, but can be estimated from the data. The estimate of $\sigma$ is known as residual standard error, and is given by the formula; 
$RSE = \sqrt{RSS/(n-2)}$. 

**pvalue** : probability of obtaining the observed results given the null hypothesis

### 3. Hypothesis Testing
Standard errors can be used to perform hypothesis tests on the coefficients. The most common hypothesis test involves testing the null hypothesis of 

$H_0$ : There is no relationship between $X$ and $Y$ ($\beta_1 = 0$) 

$H_1$ : There is some relationship between $X$ and $Y$ ($\beta_1 != 0$) 

To test the null hypothesis, we need to determine
whether $\hat{\beta_1}$, our estimate for β1, is sufficiently far from zero that we can
be confident that β1 is non-zero. How far is far enough? This of course
depends on the accuracy of $\hat{\beta_1}$ —that is, it depends on $SE(\hat{\beta_1})$. If $SE(\hat{\beta_1})$ is
small, then even relatively small values of $\hat{\beta_1}$ may provide strong evidence that β1 = 0, and hence that there is a relationship between X and Y . In contrast, if $SE(\hat{\beta_1})$ is large, then $\hat{\beta_1}$ must be large in absolute value in order
for us to reject the null hypothesis. In practice, we compute a t-statistic, t-statistic
given by

$$
\begin{align*}
\LARGE t = \frac{\hat{\beta_1}-0}{SE(\hat{\beta_1})}
\end{align*}
$$

This measure the number of standard deviations that beta_1_hat is away from 0. If there really is no relationship between x and y, then we expect that will have a t distribution with n-2 degrees of freedom. The t distribution has a bell shape and for values of n greater than approximately 30 it is quite similar to the normal distribution.

We interpret the p value as follows: a small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance, in the absence of any real association between the predictor and the response. Hence, if we see a small p-value, then we can infer that there is an association between predictor and the response. We reject the null hypothesis - that is , we declare a relationship to exist between x and y- if the p-value is small enough. 

### 4. Assessing the accuracy of the Model
The quality of a linear regression fit is typically assessed using two related quantities; the residual standard error(RSE) and the $R^2$ statistic.
Due to the presence of the error term, even if we knew the true regression line, we would not be able to perfectly predict Y from X. The RSE is an estimate of the standard deviation of the $\epsilon$. Roughly speaking, it is the average amount that the response will deviate from the true regression line. It is computed using the formua; 

$$
\begin{align*}
\LARGE RSS = \sqrt{\frac{1}{n-2}RSE} = \sqrt{\frac{1}{n-2} \sum_{i=1}^{n} (y_i-\hat{y_i})^2}
\end{align*}
$$

### 5. R2 Statistic

RSE provides an absolute measure of lack of fit of the model. But sicne it is measured in the units of y, it is not always clear what constitutes a good RSE. The $R^2$ statistic takes the form of a proportion- the proportion of variance explained - and so it always takes on a value between 0 and 1 and is independent of the scale of y. 

$$
\begin{align*}
\LARGE R^2 = \frac{TSS-RSS}{TSS} = 1- \frac{RSS}{TSS}
\end{align*}
$$
where $TSS = \sum(y_i-\bar{y})^2$ is the total sum of squares and RSS is defined. TSS measures the total variance in the response y, and can be thought of as the amount of variability inherent in the response before the regression is performed.In contrast, RSS measures the amount of variability that is left unexplained after performing the regression. R2 measures the proportion of variability in y that can be explained using x. 

### Correlation and Covariance
$$
\begin{align*}\Large
\LARGE Cor(X,Y) = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}
\end{align*}
$$

$$
\begin{equation}
\LARGE cov(X, Y) = \frac{\sum{(X_i - \bar{X})(Y_i - \bar{Y})}}{N}
\end{equation}
$$

$$
\begin{equation}
\LARGE r = \frac{cov(X, Y)}{\sqrt{var(X) \cdot var(Y)}}
\end{equation}
$$

# Multiple Linear Regression - Matrix Formulation 
https://online.stat.psu.edu/stat462/node/132/

$$
\begin{align*}\large X^{'}X=\begin{bmatrix} 1 & 1 & \cdots & 1\\  x_1 & x_2 & \cdots & x_n \end{bmatrix}\begin{bmatrix} 1 & x_1\\  1 & x_2\\  \vdots &  x_n\\   1&    \end{bmatrix}=\begin{bmatrix} n & \sum_{i=1}^{n}x_i \\  \sum_{i=1}^{n}x_i  & \sum_{i=1}^{n}x_{i}^{2} \end{bmatrix}
\end{align*}
$$

$$
\begin{align*}\large b=\begin{bmatrix} b_0\\  b_1\\  \vdots\\  b_{p-1} \end{bmatrix}=(X^{'}X)^{-1}X^{'}Y\end{align*}
$$

When we perform multiple linear regression, we usually are interested in answerng few important question. 
 * Is at least one of the predictors $X_1,X_2,...,X_p$ useful in predicting the response? 
 * Do all the predictors help to explain $Y$, or is only a subset of the predictors useful? 
 * How well does the model fit the data?
 * Given a set of predictor values, what reponse value should we predict, and how accurate is our prediction? 

** Is there a relationship between the response and predictors? 
Recall that in the simple linear regression setting, in order to determine whether there is a relationship between the response and the predictor we can simply check whether $\beta_1=0$. In the multiple linear regression with p predictors, we need to ask whether all of the regression coefficients are zero, ie, whether $ \beta_1 = \beta_2 = ... = \beta_p = 0$. We test the null hypothesis, 

$H_o : \beta_1 = \beta_2 = ... = \beta_p = 0$ versus the alternative 

$H_a :$ at least one $\beta_j$ is non-zero. 

This hypothesis is performed by computing the F-statistic. 

$$
\begin{align*}
\Large F = \frac{(TSS-RSS)/p}{RSS/(n-p-1}
\end{align*}
$$

IF f statistic takes a value close to 1, then we cannot expect any relationship between the response and predictors. Larger the f value higher the the evidence against the null hypothesis.

Sometimes, we want to test that a particular subset of q of the coefficents are zero. In this case we fit two model and compute their respective RSS value(with and without that particular subset). We use the following f-statistic to test the hypothesis. 

$$
\begin{align*}
\Large F=\frac{(RSS_0 - RSS)/q}{RSS/(n-p-1}
\end{align*}
$$


The above table provide information about whether each individual predictor is related to the response, after adjusting for the other predictors. It turns out that each of these are exactly equivalent to the F-test that omits that single variable from the model, leaving all the others in(q=1). So it reports the partial effect of adding that variable to the model.
Giving these individual p values for each variable, why do we need to look at the overall F-statistic? After all, it seems likely that if any one of the p-values for the individual variables is very small, then at least one of the predictors is related to the response. However, this logic is flawed, especially when the number of predictors p is large.
lets take p = 100, In this case, we expect to see approximately five small p-values even in the absense of any true association between the predictors and the response. F statistic does not suffer from this problem because it adjusts for the number of predictors. Hence if $H_0$ is true, there is only 5% chance that the F-statistic will result in a p-value below 0.05, regardless of the number of predictors or the number of observations. 

# Deciding on Important Variables

It is possible that all of the predictors are associated with the response,
but it is more often the case that the response is only related to a subset of
the predictors. The task of determining which predictors are associated with
the response, in order to fit a single model involving only those predictors,
is referred to as variable selection.

To determine which model is best, we use various statistics such as Mallow's Cp, Akaike information criterion, Bayesian information criterion(BIC) and adjusted R2. 

If we take all the possible combinations of p variables, then $2^p$ subsets are there. Checking these many subsets is practically impossible. 

 * Forward Selection
 * Backward Selection 
 * Mixed Seection 
 
Recall that in simple regression, R2 is the square of the correlation of the
response and the variable. In multiple linear regression, it turns out that it
equals $Cor(y, \hat{y})^2$, the square of the correlation between the response and the fitted linear model;

It turns out that $R^2$ will always incrase when more variables are added to the model, even if those variables are only weakly associated with the response. To solve this issue we have adjusted R2. 

$$
\begin{align*}
\Large Adjusted R^2 = 1 - \frac{(1-R^2)(N-1)}{N-p-1}
\end{align*}
$$

# Confidence intervals and prediction intervals

A confidence interval provides a range of values within which you can expect the true population parameter to fall. 


# Diagnosting Problems in the Regression 

1. No multicollinearity 
2. No auto correlation 
3. No heteroskedasticity. 
4. Normality 
5. Exogenity: independent variables are not correlated with the error term. 
              it arises due to the omission of explanatory variables in the regression.  
6. linear in paramters 

Note : (error terms should be 
        idependently (no autocorrelation)  and 
        identically(normality) distributed with 
        constant variance ( homoskedastic) 
        and zero mean) 


## ols violations 

### 1. multicollinearity 

Two or more independent explanatory variables have high correlation among them. 
if we have this problem, the we cannot estimate the effect of each predictor variable on the dependent variable.
multicollinearity increases the variance of the coefficient, this further increases the width of confidence interval. 

we can combine both variables, or use alternate specifications. 

* conventional diagoniss is by using vif ( variance inflation factor)
it assesses how much variance of an estimated regression coefficient increases if predictors in a model are correlated. 

vif(beta1)  = 1/ (1-r2) , tolerance = 1/vif ,  where r2 refers to the multiple correlation coefficient vetween x1 and 
other predictor variables.
if vif > 5 then it refers to the presence of multicollinearity. 

warnings signs of multicollinearity: 
    * high r2 but insignificant 
    * coefficient are opposite signs of their expected. 
    * add or remove one fetaure the regression coefcient changes dramatically. 
    * add or delete observation the regression coefficients may change substantially. 
    



### 2. Heteroskedasticity : violation of constant variance of error term. 

* heteroskedasticity refers to situvations where the variance of the residuals is unequal over 
a range of measured values. when running a regression analysis, heteroskedasticity results in an unqual scatter 
of the residuals. This is the violation of the assumption of constant variance. 

* we can analyse this from the graph of residuals. 
* formally we diagnose this using breusch-pagan test. 
    h0: the residuals are homoskedastic 
    ha: the residuals are heteroskedastic. 
* remedial measures
    *alternate methods of estimation 
    * deflating data ( log transformation, ) 
    * robust errors 



### 3. Non-normality: violationo of identical distribution of error terms

* calculation of confidence intervals and various significance tests for coefficients are all
 based on the assumptions of normally distributed errors. least sqaure estimators are 
 unbiased and mvue estimator. These properties hold even if the errors do not have the 
 normally distribution provided other assumptions are satisfied. 
 
* in order to get the confidence intervals for alpha and beta and test any hypothesis about
 alpha and beta we need the normality of errors assumption.
  
* detection is by creating QQ Plot. ========
 it is a scatter plot created by plotting two sets of quantiles against one another.
 one drawn form a theoretical normal distribution and the other drawn form the residuals of the model.  
 if both sets of quantiles come from the same distribution, the points will form a roughly straight line. 
 Qantiles : points in your data below which a certain proportion of your data falls. 

* identify the source of non-normality : outliers and non-normal distribution
* respecification of model( alternate functional)  
* logs 
* omitted explanotory variables 

* box-cox transformation: transformation of non-normal dependent variable into normalshape. at the core of the box cox transformation is the exponent lambda which varies from -5 to 5. all the values of lambda are considered and optimal vlaue for your data is selecteed. the optimal vlaue is the one which results in the best approximatio of a normal distribution curve.if the lambda is 1 then no transformation is required. if lambda is zero then log(y). 


### 4. Serial correlation or auto correlation: violation of independently distributed error terms

relationship between a given variable and a lagged version of itself over various
time intervals. ( usually in time seies data) 
presence of seriel correlation means that the errors are not independenly distrubuted.
thus the error term at t is correlated to error term at some other perios say t-1 , t-2, ... t-k. 
seial correlation is due to the correlation of omitted variable that the error term captures.
serial correlation leads to inefficient ols estimators and we can not rely on test of significance. 
	
auto correlation diagonosis is through durbin watson test. 
	dw = 2(1-corr) 
	h0: no first order auto correlation 
	h1: there exist first order auto correlation 
	
durbin watson statistic ranges from 0 to 4 , where 
		* 2 is no autocorrelation 
		* 0-2 positive auto correlation 
		* 2-4 negative auto correlation 
		* note: 1.5 to 2.5  is relatively normal. 		
		To test for positive autocorrelation at significance level α (alpha), the test statistic 
		DW is compared to lower and upper critical values:

 * If DW < Lower critical value: There is statistical evidence that the data is positively autocorrelated
 * If DW > Upper critical value: There is no statistical evidence that the data is positively correlated.
 * If DW is in between the lower and upper critical values: The test is inconclusive.	
 	
 * other tests are durbin h test, durbin alternate test, lagrange multiplier test, 	
 
 * remedial measures ( using robust standered errors) 
		
		
### leverage / standardizes residuals / cooks distance 


Measures how far away data points are from the other observations. it determines the stregnth of
 sample value on the prediction. 

standardized residuals: meaure of the strength of the difference between observed and expected values. 
if an outlier is significant it will produce substantial changes in the regression equation estimates. 

Leverage is the distance from the mass  center of the data 
cook's distance is an overall measure of influence of an observation. 
Points with high leverage may be influential: that is, deleting them would change the model a lot. 
For this we can look at Cook’s distance, which measures the effect of deleting a point on the combined parameter vector
. Cook’s distance is the dotted red line here, and points outside the dotted line have high influence. 
In this case there are no points outside the dotted line. 

		
		
# ANALYSIS OF RESIDUALS PLOTS

### 1. Analysis of Residuals vs Fitted 

  This plot shows if residuals have non-linear patterns. There could be a non-linear relationship between predictor
  variables and an outcome variable and the pattern could show up in this plot if the model doesn't capture the 
  non-linear relationship. If you find equally spread residuals around a horizontal line without a specific pattern,
  that is a good indication that  you don't have non-linear relationships. 
	
### 2. Analysis of Normal Q-Q plot 
  This plot shows if residuals are normally distributed. Do residuals follow a straight line well or do they deviate 
  severely?. It's good if residuals  are lined well on the straight dashed line. if the qq plot looks curved then the 
  residuals are skewed in one direction.  if you have heavy tails then also we have a non -normal distribution 
  flatter tails. we need normality assumption for model hypothesis tests all of this test assume that residuals 
  are normally distributed. 		
		
###  3. Analysis of scale-location plot 
  It’s also called Spread-Location plot. This plot shows if residuals are spread equally along the ranges of predictors.
   This is how you can check the assumption of equal variance (homoscedasticity). It’s good if you see a horizontal 
   line with emqually (randomly) spread points.
   
###  4. Residuals vs leverage
  Unlike the other plots, this time patterns are not relevant. We watch out for outlying values at the upper right corner 
  or at the lower right corner. Those spots are the places where cases can be influential against a regression line. 
  Look for cases outside of a dashed line, Cook’s distance. When cases are outside of the Cook’s distance
  (meaning they have high Cook’s distance scores), the cases are influential to the regression results. 
  The regression results will be altered if we exclude those cases.		
				


















