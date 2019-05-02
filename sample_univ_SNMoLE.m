function [y, klas, stats, Z] = sample_univ_SNMoLE(Alphak, Betak, Sigmak, Lambdak, x)%, n)
% draw samples from a skew-normal mixture of linear experts model
%
%
%
% X : covariates
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


n = length(x);


p = size(Betak,1)-1;
q = size(Alphak,1)-1;
K = size(Betak,2);

% construct the regression design matrices
XBeta = designmatrix_Poly_Reg(x,p); % for the polynomial regression
XAlpha = designmatrix_Poly_Reg(x,q); % for the logistic regression


y = zeros(n,1);
Z = zeros(n,K);
klas = zeros(K,1);


Deltak = Lambdak./sqrt(1+Lambdak.^2);

%calculate the mixing proportions piik:
Piik = multinomial_logit(Alphak,XAlpha);
for i=1:n
    Zik = mnrnd(1,Piik(i,:));
    %
    muk = XBeta(i,:)*Betak(:,Zik==1);
    sigmak = Sigmak(Zik==1);
    lambdak = Lambdak(Zik==1);
    % sample a skew-normal variable with the parameters of component k
    y(i) = sample_univ_SN(muk,sigmak,lambdak);
    %
    Z(i,:) = Zik;
    zi = find(Zik==1);
    klas(i) = zi;
    %
end

% Statistics (means and variances)

% E[yi|zi=k]
Ey_k = XBeta*Betak + ones(n,1)*( sqrt(2/pi)*Deltak.*Sigmak );
% E[yi]
Ey = sum(Piik.*Ey_k,2);
% Var[yi|zi=k]
Vary_k = (1 - (2/pi)*(Deltak.^2)).*(Sigmak.^2);
% Var[yi]
Vary = sum(Piik.*(Ey_k.^2 + ones(n,1)*Vary_k),2) - Ey.^2;


stats.Ey_k = Ey_k;
stats.Ey = Ey;
stats.Vary_k = Vary_k;
stats.Vary = Vary;


