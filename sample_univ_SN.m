function [y] = sample_univ_SN(mu, sigma, lambda, n)
% draw samples from a univariate skew-normal distribution
%
%
% sigma : standard deviation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin==3, n=1; end


delta = lambda./sqrt(1 + lambda.^2);

% %% non-vectorial version
% y = zeros(n,1);
% for i=1:n
%     %% by using the hierarchical representation of the skew-normal
% %     Ui = normrnd(0,sqrt(sigma2));
% %     y(i) = normrnd(mu + delta * abs(Ui), sqrt((1-delta^2))*sigma);
%
%     %% or by equivalence by using the stochastic representation of the skew-normal
%     Ui = normrnd(0, sigma);
%     Ei = normrnd(0,sigma);
%     y(i) = mu + delta * abs(Ui) + sqrt(1-delta^2)*Ei;
% end

%% vectorial version
%% by using the hierarchical representation of the skew-normal
% U = normrnd(0, sigma, n, 1);
% y = normrnd(mu + delta * abs(U), sqrt((1-delta^2))*sigma);

%% or by equivalence by using the stochastic representation of the skew-normal
U = normrnd(0, sigma, n, 1);
E = normrnd(0, sigma, n, 1);
y = mu + delta * abs(U) + sqrt(1-delta^2)*E;
