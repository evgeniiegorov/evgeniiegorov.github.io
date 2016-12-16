---
title: "Why Variational Bayes is Variational?"
categories:
  - bayesian
tags:
  - bayesian
  - variational
---

MCMC for West Cost, Variational Bayes for East Cost. So, I prefer to use the variational approach in my research over sampling. In some machine learning courses and books (as Bishop "Pattern Recognition") it looks like number of algebraic tricks. I wonder, where does real VARIATIONAL, as we know it from real analysis, hide in all it? Also, I found this way more forward and clear for me.

Consider

+ $X$ as observed data
+ $Z$ latent variables
+ $\theta$ other parameters 

Our model:

$$
p(X, Z, \theta) = p(X|Z)p(Z|\theta)p(\theta)
$$

Our far goal:

$$
p(Z,\theta|X) = \dfrac{p(X, Z, \theta)}{\int p(X, Z, \theta) dZd\theta}
$$


Most hard part is to deal with $\int p(X, Z, \theta) dZd\theta = p(X)$. Let's optimise its lower bound.

# Exact solution

$$
\ln p(X) = \ln \int p(X, Z, \theta) dZd\theta = \ln \int q(X, Z)p(X, Z, \theta)\dfrac{1}{q(X, Z)} dZd\theta = \\
= \ln\left(\mathbb{E}_{q(\theta,Z)}\dfrac{ p(X, Z, \theta)}{q(X, Z)}\right)\geq \{\text{Concave of log}\} \geq \mathbb{E}_{q(\theta,Z)}\ln\dfrac{ p(X, Z, \theta)}{q(X, Z)} = \\
= \mathbb{E}_q \ln p(X,Z|\theta) + \mathbb{E}_q \ln p(\theta) - \mathbb{E}_q\ln q(Z, \theta)
$$

Consider 
$$\mathcal{F}(q) = \mathbb{E}_q \ln p(X,Z|\theta) + \mathbb{E}_q \ln p(\theta) - \mathbb{E}_q\ln q(Z, \theta)$$

Than we obtain optimization problem with constraints:

$$
\max\mathcal{L}(q,\lambda) = \mathcal{F}(q) + \lambda\left(\int q(Z,\theta)dZd\theta - 1 \right) = \\
= \mathbb{E}_q \ln p(X,Z|\theta) + \mathbb{E}_q \ln p(\theta) - \mathbb{E}_q\ln q(Z, \theta) + \lambda\left(\int q(Z,\theta)dZd\theta - 1 \right)
$$

FOC:

$$
\dfrac{\delta}{\delta q}\mathcal{L}(q,\lambda) = \dfrac{\delta}{\delta q}\mathcal{F}(q)+\lambda = \ln p(X,Z,\theta)  - \left(\ln q(Z,\theta) + 1\right) + \lambda = 0
$$


$$
\dfrac{\partial}{\partial \lambda}\mathcal{L}(q,\lambda) = \int q(Z,\theta)dZd\theta - 1 = 0
$$


From first condition:
    
$$
\ln p(X,Z,\theta) = \ln q(Z,\theta) + 1 - \lambda \\
q(Z,\theta) = p(X,Z,\theta)\exp(\lambda - 1) \\
$$

Then from second condition:

$$
\exp(\lambda - 1)\int p(X,Z,\theta) dZd\theta = 1 \\
\lambda = -\ln \int p(X,Z,\theta)dZd\theta + 1
$$

Finally:

$$
q(Z,\theta) = \dfrac{p(X,Z,\theta)}{\int p(X, Z,\theta)dZd\theta}
$$

We end up with exact solution. Now we will follow same pipeline, but use mean field approximation.

# Mean field approximation

Sometimes (be honest, almost surely) we can’t compute, $\int p(X, Z,\theta)dZd\theta$ directly. Then we have restrict to set of functions. Thus, we condiser $q(Z,\theta) = q_z(Z)q_{\theta}(\theta)$.

Then:

$$
\mathcal{F}(q) = \mathbb{E}_q \ln p(X,Z|\theta) + \mathbb{E}_q \ln p(\theta) - \mathbb{E}_q\ln q(Z, \theta) = \\
= \mathbb{E}_q \ln p(X,Z|\theta) + \mathbb{E}_{q_{\theta}} \ln p(\theta) - \mathbb{E}_{q_{\theta}}\ln q_{\theta}
(\theta) - \mathbb{E}_{q_{z}} \ln q_{z}(Z)
$$

As earlier FOC:


$$
\dfrac{\delta}{\delta q_{\theta}}\mathcal{L} = \mathbb{E}_{q_{z}}
\ln p(X,Z|\theta) + \ln p(\theta) - (\ln q_{\theta}(\theta)+1) + \lambda_{\theta} = 0 
$$

$$
\dfrac{\delta}{\delta q_{z}}\mathcal{L} = \mathbb{E}_{q_{\theta}}
\ln p(X,Z|\theta) - (\ln q_{z}(Z)+1) + \lambda_z = 0 
$$

$$
\dfrac{\partial}{\partial \lambda_{\theta}}\mathcal{L} = \int q_{\theta}(\theta)d\theta - 1 = 0
$$

$$
\dfrac{\partial}{\partial \lambda_{z}}\mathcal{L} = \int q_{z}(Z)dZ - 1 = 0
$$

Then, for $q_\theta$:

$$
q_{\theta}(\theta) = p(\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\theta)-1+\lambda_{\theta})
$$

$$
\lambda_{\theta} = 1 - \ln\int p(\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\theta)d\theta
$$

Finally:

$$
q_{\theta}(\theta) = \dfrac{p(\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\theta)}{\int p(\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\theta)d\theta}
$$

By symmetry:

$$
q_{z}(Z) = \dfrac{\exp(\mathbb{E}_{q_{\theta}}\ln p(X,Z|\theta))}{\int\exp(\mathbb{E}_{q_{\theta}}\ln p(X,Z|\theta)dZ}$$

And that's it!
