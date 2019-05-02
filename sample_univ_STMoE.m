function [y, klas, stats, Z] = sample_univ_STMoE(Alphak, Betak, Sigmak, Lambdak, Nuk, x)
% draw samples from a univariate skew-t mixture
%
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
    nuk = Nuk(Zik==1);
    % sample a skew-t variable with the parameters of components k
    y(i) = sample_univ_ST(muk, sigmak, nuk, lambdak);
    
    %
    Z(i,:) = Zik;
    zi = find(Zik==1);
    klas(i) = zi;
    %
end

% %E[yi|zi=k]
%
% Ey_k = (XBeta*Betak);% .* (ones(n,1)*(sqrt(Nuk/2).*((1 - 3./(4*Nuk-1)).^(-1))));
%
% % E[yi]
% Ey = sum(Piik.*Ey_k,2);
%
% % Statistics (means, variances)
% Xi_nuk = sqrt(Nuk/pi).*((1- 3./(4*Nuk-1)).^(-1)).*sqrt(2./Nuk);%%%%(gamma(Nuk/2 +1/2))./(gamma(Nuk/2));
% % E[yi|zi=k]
% Ey_k = XBeta*Betak + ones(n,1)*( Deltak.*Sigmak.*Xi_nuk );
% % E[yi]
% Ey = sum(Piik.*Ey_k,2);
% % Var[yi|zi=k]
% Vary_k = (Nuk./(Nuk-2) - (Deltak.^2).*(Xi_nuk.^2)).*(Sigmak.^2);
%
% % Var[yi]
% Vary = sum(Piik.*(Ey_k.^2 + ones(n,1)*Vary_k),2) - Ey.^2;


% Statistics (means, variances)
Xi_nuk = sqrt(Nuk/pi).*(gamma(Nuk/2 -1/2))./(gamma(Nuk/2)); 
% E[yi|zi=k]
Ey_k = XBeta*Betak + ones(n,1)*(sigmak.*Deltak.*Xi_nuk);

% E[yi]
Ey = sum(Piik.*Ey_k,2);

% Var[yi|zi=k]
Vary_k = (Nuk./(Nuk-2) - (Deltak.^2).*(Xi_nuk.^2)).*(Sigmak.^2);
%ones(n,1)*((Nuk./(Nuk-2)).*(Sigmak.^2)) - Ey_k.^2;
%(XBeta*Betak).^2 + 2*XBeta*Betak.*(ones(n,1)*(Deltak.*Sigmak.*Xi_nuk)) + ones(n,1)*((Nuk./(Nuk-2)).*(Sigmak.^2));

% Var[yi]
Vary = sum(Piik.*(Ey_k.^2 + ones(n,1)*Vary_k),2) - Ey.^2;

stats.Ey_k = Ey_k;
stats.Ey = Ey;
stats.Vary_k = Vary_k;
stats.Vary = Vary;

