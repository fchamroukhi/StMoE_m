function [stme_pdf, piik_fik] = univ_STMoE_pdf(y, Alphak, Betak, Sigmak, Lambdak, Nuk, XBeta, XAlpha)
%
%
%
%
% returs the mixture density and the components densities (in log) for the
% skew t mixture of experts (STMoE)
%
% C By Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size(y,2)~=1, y=y'; end

n = length(y);
K = length(Sigmak);

Piik = multinomial_logit(Alphak,XAlpha);

piik_fik = zeros(n,K);
Dik = zeros(n,K);
Mik = zeros(n,K); 
for k = 1:K
    Dik(:,k) = (y - XBeta*Betak(:,k))/Sigmak(k);
    Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2));
    % piik*STE(.;muk;sigma2k;lambdak)
    %weighted skew t linear expert likelihood
    piik_fik(:,k) = Piik(:,k).*(2/Sigmak(k)).*tpdf(Dik(:,k), Nuk(k)).*tcdf(Mik(:,k), Nuk(k)+1);
end

stme_pdf = sum(piik_fik,2);% skew-t mixture of experts density
% log_stme_pdf=log(stm_pdf);