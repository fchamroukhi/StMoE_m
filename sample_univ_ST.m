function [y] = sample_univ_ST(mu, sigma, nu, lambda, n)
% draw samples from a univariate skew-t distribution
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin ==4, n=1; end

%% non-vectorial version
% y = zeros(n,1);
% for i=1:n
%     %% by using the hierarchical representation of the skew-t distribution
% %     delta = lambda./sqrt(1+lambda.^2);
% %     Wi = gamrnd(nu/2,2/nu);% or by equivalance : chi2rnd(nu)/nu;%
% %     Ui = normrnd(0,sqrt(sigma^2/Wi));
% %     y(i) = normrnd(mu + delta * abs(Ui), sqrt((1-delta^2)*sigma^2/Wi));
%     %% or by equivalance by using the stochastic representation of the
%     % skew-t distribution :
%     % a stanard skew-normal variable with parameter lambda
%     Ei = sample_univ_SN(0, 1, lambda);
%     % a gamma variable
%     Wi = gamrnd(nu/2,2/nu); %chi2rnd(nu)/nu;
%     % a skew-t variable
%     y(i) = mu + sigma*Ei/sqrt(Wi);
%     %% another way :
% %     delta = lambda./sqrt(1+lambda.^2);
% %     Ui = normrnd(0, sigma)*sqrt(nu/(2*randg(nu/2,1)));
% %     y(i) = normrnd(mu + delta * abs(Ui), sqrt(1-delta^2)*sigma);
% end

%% vectorial version
%% by using the hierarchical representation of the skew-t distribution
%     delta = lambda/sqrt(1+lambda^2);
%     W = gamrnd(nu/2,2/nu, n, 1);% or by equivalance : chi2rnd(nu, n, 1)/nu;%
%     U = normrnd(0,sqrt(sigma^2/W), n, 1);
%     y  = normrnd(mu + delta * abs(U), sqrt((1-delta^2)*sigma^2./W));
%% or by equivalance by using the stochastic representation of the
% skew-t distribution :
% a stanard skew-normal variable with parameter lambda
E = sample_univ_SN(0, 1, lambda, n);
% a gamma variable
W = gamrnd(nu/2,2/nu); %chi2rnd(nu, n, 1)/nu;
% a skew-t variable
y = mu + sigma*E./sqrt(W);
%% another way :
%     delta = lambda/sqrt(1+lambda^2);
%     U = normrnd(0, sigma, n, 1).*sqrt(nu./(2*randg(nu/2, n, 1)));
%     y = normrnd(mu + delta * abs(U), sqrt(1-delta^2)*sigma);
end