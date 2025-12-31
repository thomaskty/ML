
# Logistic Regression
* Discriminative probabiliy classifier
* Models the probability of a binary outcome as a smooth function of predictors, constrained between 0 and 1.

$$
\begin{equation}
Y \in \{0,1\}
P(Y=1) =p
P(Y=0) = 1-p
P(Y=y) = p^y(1-p)^{1-y}
\end{equation}
$$
Why is log-odds the natural parameter of bernoulli distribution? 
 * We start with bernoulli pdf 
 * Write it in exponential form 
 * The coefficient multiply $y$ must be the natural parameter
 * so log-odds is not a design choice, it emerges.

Consider; $p^y(1-p)^{1-y}$, lets write it in exponential form; 

$$
\begin{align*}
p^y(1-p)^{1-y} &= exp(y\log{p} + (1-y) \log{1-p}))
\\ &= exp(y\log{p} + \log(1-p) - y\log{1-p})
\\ &= exp(y\log{\frac{p}{1-p}}+ \log{1-p})
\end{align*}
$$

Canonical exponential form is as follows ; 
$$
\begin{align*}
f(y|\theta) = exp(y\theta - A(\theta))
\end{align*}
$$
So, based on our exponential form of bernoulli, 
$$
\begin{align*}
\theta = \log{\frac{p}{1-p}}
\\ -A(\theta) = \log{1-p}
\end{align*}
$$
So, $\log{\frac{p}{1-p}}$ is a natural parameter. 

### Introducing Covariates: Conditional Bernoulli
Once predictors exist, we are no longer modeling p, but

$$
\begin{align*}
p(x) = P(Y=1|X=x)
\end{align*}
$$

So, the natural parameter becomes conditional; 
$$
\begin{align*}
\theta(x) = \log{(\frac{p(x)}{1-p(x)})}
\end{align*}
$$

**We make one structural assumption: The natural parameter depends linearly on predictors**

$$
\begin{align*}
\theta(x) = \beta_0 + \beta^Tx
\end{align*}
$$

* $\theta \in (-\infty,+\infty)$
* Linear structure
* Linear natural parameter implies convex log-likelihood

When we plug back the theta value into $p(x)$;

$$
\begin{align*}
p(x) = \frac{1}{1+e^{-(\beta_0+\beta^Tx)}}
\end{align*}
$$

### Other linking functions
* Probit - Normal CDF
* Complementary log-log
* Cauchit
* Only logit is the canonical link for Bernoulli
* Canonical link -> best statistical properties.

### Likelihood, Log-Likelihood, and Cross-Entropy Loss
For a single observation $(x_i, y_i)$ with $y_i \in \{0,1\}$, the Bernoulli probability mass function is:

$$
\begin{align*}
P(Y_i = y_i \mid X_i = x_i)
&= p_i^{y_i}(1-p_i)^{1-y_i}
\end{align*}
$$

where  
$p_i = P(Y_i=1 \mid X_i=x_i)$.

Assuming **conditional independence** of observations given $X$, the likelihood for the full dataset $\{(x_i,y_i)\}_{i=1}^n$ is the product of individual likelihoods:

$$
\begin{align*}
L(\beta)
&= \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
\end{align*}
$$

Here, the probabilities $p_i$ are modeled using the logistic function:

$$
\begin{align*}
p_i
&= \frac{1}{1 + \exp(-\eta_i)} \\
\eta_i
&= \beta_0 + \beta^T x_i
\end{align*}
$$

Since the log function is monotonic, maximizing the likelihood is equivalent to maximizing the log-likelihood.

$$
\begin{align*}
\ell(\beta)
&= \log L(\beta) \\
&= \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]
\end{align*}
$$

This is the **log-likelihood** of logistic regression.
In optimization, it is common to convert a maximization problem into a minimization problem. We therefore define the **negative log-likelihood (NLL)**:

$$
\begin{align*}
\mathcal{L}(\beta)
&= -\ell(\beta) \\
&= - \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]
\end{align*}
$$

The goal of logistic regression is:

$$
\begin{align*}
\hat{\beta}
= \arg\min_{\beta} \; \mathcal{L}(\beta)
\end{align*}
$$


The negative log-likelihood above is known in machine learning as the **binary cross-entropy loss** (or log loss).

For a single observation, the loss is:

$$
\begin{align*}
\text{BCE}(y_i, p_i)
&= - \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]
\end{align*}
$$

For the full dataset:

$$
\begin{align*}
\text{BCE}(\beta)
&= \sum_{i=1}^{n} \text{BCE}(y_i, p_i)
\end{align*}
$$

- This loss arises **directly from maximum likelihood estimation**, not as a heuristic.
- Thus, minimizing binary cross-entropy is equivalent to **maximum likelihood estimation for a Bernoulli model with a logistic link**.

Thus, Logistic regression estimates parameters $\beta$ by:

1. Modeling $P(Y=1 \mid X=x)$ via the logistic (sigmoid) function  
2. Writing the Bernoulli likelihood for observed labels  
3. Maximizing the log-likelihood (or equivalently, minimizing binary cross-entropy)
