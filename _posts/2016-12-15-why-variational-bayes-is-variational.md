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
+ $\t\theta$ other parameters 

Our model:

$$
p(X, Z, \t\theta) = p(X|Z)p(Z|\t\theta)p(\t\theta)
$$

Our far goal:

$$
p(Z,\t\theta|X) = \dfrac{p(X, Z, \t\theta)}{\int p(X, Z, \t\theta) dZd\t\theta}
$$


Most hard part is to deal with $\int p(X, Z, \t\theta) dZd\t\theta = p(X)$. Let's optimise its lower bound.

# Exact solution

$$
\ln p(X) = \ln \int p(X, Z, \t\theta) dZd\t\theta = \ln \int q(	\theta, Z)p(X, Z, \t\theta)\dfrac{1}{q(	\theta, Z)} dZd\t\theta = \\
= \ln\left(\mathbb{E}_q(\t\theta, Z){q(\t\theta,Z)}\dfrac{ p(X, Z, \t\theta)}{q(	\theta, Z)}\right)\geq \{\text{Concave of log}\} \geq \mathbb{E}_{q(\t\theta,Z)}\ln\dfrac{ p(X, Z, \t\theta)}{q(	\theta, Z)} = \\
= \mathbb{E}_q \ln p(X,Z|\t\theta) + \mathbb{E}_q \ln p(\t\theta) - \mathbb{E}_q\ln q(Z, \t\theta)
$$

Consider 
$$\mathcal{F}(q) = \mathbb{E}_q \ln p(X,Z|\t\theta) + \mathbb{E}_q \ln p(\t\theta) - \mathbb{E}_q\ln q(Z, \t\theta)$$

Than we obtain optimization problem with constraints:

$$
\max\mathcal{L}(q,\lambda) = \mathcal{F}(q) + \lambda\left(\int q(Z,\t\theta)dZd\t\theta - 1 \right) = \\
= \mathbb{E}_q \ln p(X,Z|\t\theta) + \mathbb{E}_q \ln p(\t\theta) - \mathbb{E}_q\ln q(Z, \t\theta) + \lambda\left(\int q(Z,\t\theta)dZd\t\theta - 1 \right)
$$

FOC:

$$
\dfrac{\delta}{\delta q}\mathcal{L}(q,\lambda) = \dfrac{\delta}{\delta q}\mathcal{F}(q)+\lambda = \ln p(X,Z,\t\theta)  - \left(\ln q(Z,\t\theta) + 1\right) + \lambda = 0
$$


$$
\dfrac{\partial}{\partial \lambda}\mathcal{L}(q,\lambda) = \int q(Z,\t\theta)dZd\t\theta - 1 = 0
$$


From first condition:
    
$$
\ln p(X,Z,\t\theta) = \ln q(Z,\t\theta) + 1 - \lambda \\
q(Z,\t\theta) = p(X,Z,\t\theta)\exp(\lambda - 1) \\
$$

Then from second condition:

$$
\exp(\lambda - 1)\int p(X,Z,\t\theta) dZd\t\theta = 1 \\
\lambda = -\ln \int p(X,Z,\t\theta)dZd\t\theta + 1
$$

Finally:

$$
q(Z,\t\theta) = \dfrac{p(X,Z,\t\theta)}{\int p(X, Z,\t\theta)dZd\t\theta}
$$

We end up with exact solution. Now we will follow same pipeline, but use mean field approximation.

# Mean field approximation

Sometimes (be honest, almost surely) we can’t compute, $\int p(X, Z,\t\theta)dZd\t\theta$ directly. Then we have restrict to set of functions. Thus, we condiser $q(Z,\t\theta) = q_z(Z)q_{\t\theta}(\t\theta)$.

Then:

$$
\mathcal{F}(q) = \mathbb{E}_q \ln p(X,Z|\t\theta) + \mathbb{E}_q \ln p(\t\theta) - \mathbb{E}_q\ln q(Z, \t\theta) = \\
= \mathbb{E}_q \ln p(X,Z|\t\theta) + \mathbb{E}_{q_{\t\theta}} \ln p(\t\theta) - \mathbb{E}_{q_{\t\theta}}\ln q_{\t\theta}
(\t\theta) - \mathbb{E}_{q_{z}} \ln q_{z}(Z)
$$

As earlier FOC:


$$
\dfrac{\delta}{\delta q_{\t\theta}}\mathcal{L} = \mathbb{E}_{q_{z}}
\ln p(X,Z|\t\theta) + \ln p(\t\theta) - (\ln q_{\t\theta}(\t\theta)+1) + \lambda_{\t\theta} = 0 
$$

$$
\dfrac{\delta}{\delta q_{z}}\mathcal{L} = \mathbb{E}_{q_{\t\theta}}
\ln p(X,Z|\t\theta) - (\ln q_{z}(Z)+1) + \lambda_z = 0 
$$

$$
\dfrac{\partial}{\partial \lambda_{\t\theta}}\mathcal{L} = \int q_{\t\theta}(\t\theta)d\t\theta - 1 = 0
$$

$$
\dfrac{\partial}{\partial \lambda_{z}}\mathcal{L} = \int q_{z}(Z)dZ - 1 = 0
$$

Then, for $q_\t\theta$:

$$
q_{\t\theta}(\t\theta) = p(\t\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\t\theta)-1+\lambda_{\t\theta})
$$

$$
\lambda_{\t\theta} = 1 - \ln\int p(\t\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\t\theta)d\t\theta
$$

Finally:

$$
q_{\t\theta}(\t\theta) = \dfrac{p(\t\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\t\theta)}{\int p(\t\theta)\exp(\mathbb{E}_{q_{z}}\ln p(X,Z|\t\theta)d\t\theta}
$$

By symmetry:

$$
q_{z}(Z) = \dfrac{\exp(\mathbb{E}_{q_{\t\theta}}\ln p(X,Z|\t\theta))}{\int\exp(\mathbb{E}_{q_{\t\theta}}\ln p(X,Z|\t\theta)dZ}$$

And that's it!
