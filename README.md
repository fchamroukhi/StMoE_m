>>>> skew-t mixture of experts (STMoE) <<<<<

STMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
heterogeneous data with the Skew-t Mixture-of-Experts (STMoE) model.

STMoE provides a flexible and robust modeling framework for heterogenous data with possibly
skewed, heavy-tailed distributions and corrupted by atypical observations. STMoE consists of a
mixture of K skew-t expert regressors network (of degree p) gated by a softmax gating network
(with regression degree q) and is represented by - The gating net. parameters $\alpha$'s of the
softmax net. - The experts network parameters: The location parameters (regression coefficients)
$\beta$'s, scale parameters $\sigma$'s, the skewness parameters $\lambda$'s and the degree of
freedom parameters $\nu$'s. STMoE thus generalises  mixtures of (normal, skew-normal, t, and
skew-t) distributions and mixtures of regressions with these distributions. For example, when
$q=0$, we retrieve mixtures of (skew-t, t-, skew-normal, or normal) regressions, and when both
$p=0$ and $q=0$, it is a mixture of (skew-t, t-, skew-normal, or normal) distributions. It also
reduces to the standard (normal, skew-normal, t, and skew-t) distribution when we only use a
single expert (K=1).

Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
illustrate the use of the model in model-based clustering of heteregenous regression data and in
fitting non-linear regression functions. Real-world data examples of tone perception for musical
data analysis, and the one of temperature anomalies for the analysis of climate change data, are
also provided as application of the model.

To run it on the provided examples, please run "main_demo_STMoE_SimulatedData.m" or "main_demo_STMoE_RealData.m"

Please cite the code and the following papers when using this code: 
- F. Chamroukhi. Skew t mixture of experts. Neurocomputing - Elsevier, Vol. 266, pages:390-408, 2017 
- F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015
 
(c) Introduced and written by Faicel Chamroukhi (may 2015)