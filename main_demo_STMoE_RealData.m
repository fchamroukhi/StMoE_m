%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% >>>> skew-t mixture of experts (STMoE) <<<<<
% 
% STMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
% heterogeneous data with the Skew-t Mixture-of-Experts (STMoE) model.
% 
% STMoE provides a flexible and robust modeling framework for heterogenous data with possibly
% skewed, heavy-tailed distributions and corrupted by atypical observations. STMoE consists of a
% mixture of K skew-t expert regressors network (of degree p) gated by a softmax gating network
% (with regression degree q) and is represented by 
% - The gating net. parameters $\alpha$'s of the softmax net.
% - The experts network parameters: The location parameters (regression coefficients) $\beta$'s,
% scale parameters $\sigma$'s, the skewness parameters $\lambda$'s and the degree of freedom
% (robustness) parameters $\nu$'s.
%
% STMoE thus generalises  mixtures of (normal, skew-normal, t, and skew-t) distributions and
% mixtures of regressions with these distributions. For example, when $q=0$, we retrieve mixtures of
% (skew-t, t-, skew-normal, or normal) regressions, and when both $p=0$ and $q=0$, it is a mixture
% of (skew-t, t-, skew-normal, or normal) distributions. It also reduces to the standard (normal,
% skew-normal, t, and skew-t) distribution when we only use a single expert (K=1).
% 
% Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
% algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
% illustrate the use of the model in model-based clustering of heteregenous regression data and in
% fitting non-linear regression functions. Real-world data examples of tone perception for musical
% data analysis, and the one of temperature anomalies for the analysis of climate change data, are
% also provided as application of the model.
% 
%% Please cite the code and the following papers when using this code: 
% - F. Chamroukhi. Skew t mixture of experts. Neurocomputing - Elsevier, Vol. 266, pages:390-408, 2017 
% - F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015
%% 
% (c) Introduced and written by Faicel Chamroukhi (may 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;


set(0,'defaultaxesfontsize',14);
%%  chose a real data and some model structure
data_set = 'Tone'; K = 2; p = 1; q = 1;
data_set = 'TemperatureAnomaly'; K = 2; p = 1; q = 1;
data_set = 'motorcycle'; K = 4; p = 2; q = 1;

%% EM options
nbr_EM_tries = 2;
max_iter_EM = 1500;
threshold = 1e-6;
verbose_EM = 1;
verbose_IRLS = 0;

switch data_set
    %% Tone data set
    case 'Tone'
        data = xlsread('data/Tone.xlsx');
        x = data(:,1);
        y = data(:,2);
        %% Temperature Anomaly
    case 'TemperatureAnomaly'
        load 'data/TemperatureAnomaly';
        x = TemperatureAnomaly(:,1);%(3:end-2,1); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
        y = TemperatureAnomaly(:,2);%(3:end-2,2); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
        %% Motorcycle
    case 'motorcycle'
        load 'data/motorcycle.mat';
        x=motorcycle.x;
        y=motorcycle.y;
    otherwise
        data = xlsread('data/Tone.xlsx');
        x = data(:,1);
        y = data(:,2);
end
figure,
plot(x, y, 'ko')
xlabel('x')
ylabel('y')
title([data_set,' data set'])

%% learn the model from the  data

STMoE =  learn_STMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);

disp('- fit completed --')

show_STMoE_results(x, y, STMoE)
% Note that as it uses the skew-t, so the mean and the variance might be not defined (if Nu <1 and or <2), and hence the
% mean functions and confidence regions might be not displayed..



