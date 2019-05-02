function solution = learn_univ_STMoE_EM(Y, x, K, p, dim_w, ...
    total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)

% Faicel CHAMROUKHI
% Mise a jour (Mai 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off


if nargin<10 verbose_IRLS = 0; end
if nargin<9  verbose_IRLS =0; verbose_EM = 0; end;
if nargin<8  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; end;
if nargin<7  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; end;
if nargin<6  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; total_EM_tries=1;end;

if size(Y,2)==1, Y=Y'; end % cas d'une courbe

[n, m] = size(Y); % n curves, each curve is composed of m observations
q = dim_w;

% construct the regression design matrices
XBeta = designmatrix_Poly_Reg(x,p); % for the polynomial regression
XAlpha = designmatrix_Poly_Reg(x,q); % for the logistic regression

XBeta  = repmat(XBeta,n,1);
XAlpha = repmat(XAlpha,n,1);


y = reshape(Y',[],1);

best_loglik = -inf;
stored_cputime = [];
EM_try = 1;
while EM_try <= total_EM_tries,
    if total_EM_tries>1, disp(sprintf('EM run n°  %d  ',EM_try)); end
    time = cputime;
    %% EM Initialisation 
    
    %1. Initialisation of Alphak's, Betak's and Sigmak's
    segmental = 0;
    
    [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y,K, XAlpha, XBeta, segmental);    
    
    if EM_try ==1, Alphak = rand(q+1,K-1);end % set the first initialization to the null vector
    
%     solution = learn_univ_NMoE_EM(Y, x, K, p, q, 1, 500, 1e-6, 1, 0);
% %     total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)
%     Alphak=solution.param.Alphak;
%     Betak =solution.param.Betak;
%     Sigmak =sqrt(solution.param.Sigmak);
%     Sigma2k = Sigmak.^2;

%     Deltak = solution.Deltak;
%     Lambdak = solution.param.Lambdak;

    % 2. Intitialization of the skewness parameter
 
    Deltak = -.9 + 1.8*rand(1,K); % tone : -.9 + .9*rand(1,K); and 5 rand for Nuk, zeros alpha and non symetri
    Lambdak = Deltak./sqrt(1-Deltak.^2);
    
    % 3. Intitialization of the degrees of freedm
    Nuk = 1 + 5*rand(1,K);
    
    % intial value for STMoE density
    [STMoE, tmp] = univ_STMoE_pdf(y, Alphak, Betak, sqrt(Sigma2k), Lambdak, Nuk, XBeta, XAlpha) ; 
    
    %%%
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    %AlphakInit = Alphak;%
    %% EM %%%%
    while ~converge && (iter< max_iter_EM)
        iter=iter+1;
        %% E-Step
        Piik = multinomial_logit(Alphak,XAlpha);
        
        piik_fik = zeros(m*n,K);
        
        Dik = zeros(m*n,K);
        Mik = zeros(m*n,K);
        Wik = zeros(m*n,K);
        %
        E1ik = zeros(m*n,K);
        E2ik = zeros(m*n,K);
        E3ik = zeros(m*n,K);
        for k = 1:K
            muk = XBeta*Betak(:,k);
            sigma2k = Sigma2k(k);
            sigmak = sqrt(sigma2k);
            Dik(:,k) = (y - muk)/sigmak;
            
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2));
            
%             A = tcdf(Mik(:,k)*sqrt((Nuk(k) + 3)/(Nuk(k) + 1)), Nuk(k) + 3);
%             B = tcdf(Mik(:,k), Nuk(k) + 1);

%  xsq = (Mik(:,k)*sqrt((Nuk(k) + 3)/(Nuk(k) + 1))).^2;
% v= Nuk(k) + 3;

% min(xsq./(v + xsq))
% %    
%  %Nuk
%  subplot(211),plot(A)
%  subplot(212),plot(B)
% subplot(313),plot(Mik(:,k))
% Nuk
% pause

% plot(xsq)
% pause
            % E[Wi|yi,zik=1]
            Wik(:,k) = ((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2)).*...
                tcdf(Mik(:,k)*sqrt((Nuk(k) + 3)/(Nuk(k) + 1)), Nuk(k) + 3)./tcdf(Mik(:,k), Nuk(k) + 1);
% Wik(:,k) = abs(Wik(:,k));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % E[Wi Ui |yi,zik=1]
            deltak = Deltak(k);
            
            E1ik(:,k) = deltak * abs(y - muk).*Wik(:,k) + ...
                (sqrt(1 - deltak^2)./(pi * STMoE)).*...
                ((Dik(:,k).^2/(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
%  E1ik(:,k) = abs( E1ik(:,k));
            % E[Wi Ui^2|yi,zik=1]
            E2ik(:,k) = deltak^2 * ((y - muk).^2).*Wik(:,k) + (1 - deltak^2)*sigmak^2 + ...
                ((deltak * (y - muk) * sqrt(1 - deltak^2))./(pi * STMoE)).*...
                (((Dik(:,k).^2)./(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %     Wik = abs(Wik);
%     %E1ik = abs(E1ik);
% %     E2ik = abs(E2ik); 
%             %alternative representation
%             % E[Ui|yi,zik=1] and E[Ui^2|yi,zik=1]
%             mu_uk = (Deltak(k)* (y - XBeta*Betak(:,k)));
%             sigma2_uk = (1-Deltak(k)^2)*Sigma2k(k);
%             sigma_uk = sqrt(sigma2_uk);
%             % E[Wi Ui |yi,zik=1] and E[Wi Ui^2|yi,zik=1]
%             TMP = (1./(pi * STMoE)).*(((Dik(:,k).^2)./(Nuk(k)*(1 - Deltak(k)^2)) + 1).^(-(Nuk(k)/2 + 1)));
%             % E[Wi Ui |yi,zik=1]
%             E1ik(:,k) = mu_uk.*Wik(:,k) + (sigma_uk/sigmak).* TMP;
%             % E[Wi Ui^2|yi,zik=1]
%             E2ik(:,k) = mu_uk.^2.*Wik(:,k) + sigma2_uk + mu_uk*(sigma_uk/sigmak).*TMP;
%  %  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
            % E[log(Wi)|yi,zik=1]
            % numerical integral computation
%             gtx = @(h) (psi((nuk+2)+2) -  psi((nuk+1)+2) - log(1+ (h.^2/(nuk+1))) + ((nuk+1)*h.^2 - nuk - 1)./((nuk+1)*(nuk+1+h.^2)))...
%                 .*tpdf(h, nuk + 1);
%             Integgtx = zeros(n,1);
%             for i=1:n
%                 Integgtx(i) = quad(gtx,-100,mik(i));
%             end
            Integgtx = 0;
            
            E3ik(:,k) = Wik(:,k) - log((Nuk(k) + Dik(:,k).^2)/2) -(Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2) + psi((Nuk(k) + 1)/2)...
                + ((Lambdak(k)*Dik(:,k).*(Dik(:,k).^2 - 1))./sqrt((Nuk(k) + 1)*((Nuk(k) + Dik(:,k).^2).^3))).*...
                tpdf(Mik(:,k), Nuk(k)+1)./tcdf(Mik(:,k), Nuk(k) + 1)...
                                + (1./tcdf(Mik(:,k), Nuk(k) + 1)).*Integgtx;             
            % piik*STE(.;muk;sigma2k;lambdak)
            %weighted skew t linear expert likelihood
            piik_fik(:,k) = Piik(:,k).*(2/sigmak).*tpdf(Dik(:,k), Nuk(k)).*tcdf(Mik(:,k), Nuk(k) + 1);
        end
        
        STMoE = sum(piik_fik,2);% skew-t mixture of experts density
        log_piik_fik = log(piik_fik);
        log_sum_piik_fik = log(sum(piik_fik,2));
        
        %E[Zi=k|yi]
        Tauik = piik_fik./(STMoE*ones(1,K));%(sum(piik_fik,2)*ones(1,K));
        
%             figure
%      plot(Tauik)
% pause     
        %% M-Step
        % updates of alphak, betak's, sigma2k's and lambdak's
        % --------------------------------------------------%
        %% CM-Step 1 update of the softmax parameters (Alphak)
        %%  IRLS for multinomial logistic regression
        res = IRLS(XAlpha, Tauik, Alphak,verbose_IRLS);
        %Piik = res.piik;
        Alphak = res.W;
        for k=1:K,
            %% CM-Step 1:
            % update the regression coefficients
%             XBetak = XBeta.*(sqrt(Tauik(:,k).*Wik(:,k))*ones(1,p+1));%[m*(p+1)]
%             yk = sqrt(Tauik(:,k).*Wik(:,k)).*y;% dimension :(nx1).*(nx1) = (nx1)
%             betak = XBetak'*XBetak\(XBetak'*(yk - Deltak(k)*(sqrt(Tauik(:,k))./Wik(:,k).*E1ik(:,k))));
%             Betak(:,k) = betak;
            %
            TauikWik = (Tauik(:,k).*Wik(:,k))*ones(1,p+1);
            TauikX = XBeta.*(Tauik(:,k)*ones(1,p+1));
            betak =((TauikWik.*XBeta)'*XBeta)\(TauikX'*(Wik(:,k).*y - Deltak(k)*E1ik(:,k)));

            
%             TauikX = XBeta.*(sqrt(Tauik(:,k)*ones(1,p+1))); 
%             WikX = XBeta.*(sqrt(Wik(:,k))*ones(1,p+1));%[m*(p+1)]            
%             betak = TauikX'*WikX\(TauikX'*(Wik(:,k).*y - Deltak(k)*E1ik(:,k)));
            Betak(:,k) = betak;
            
            % update the variances sigma2k
            Sigma2k(k)= sum(Tauik(:,k).*(Wik(:,k).*((y-XBeta*betak).^2) - 2*Deltak(k)*E1ik(:,k).*(y-XBeta*betak)...
                + E2ik(:,k)))/(2*(1-Deltak(k)^2)*sum(Tauik(:,k)));
            
            %Sigma2k(k)=.1;
            %% CM-Step 2: update the deltak (the skewness parameter)
            delta0 = Deltak(k);
%             e1ik = abs(E1ik(:,k));
%             e2ik = (E2ik(:,k));
            %
            sigmak = sqrt(Sigma2k(k));
            Dik(:,k) = (y - XBeta*Betak(:,k))/sigmak;
            
            Deltak(k) = fzero(@(delta) delta*(1-delta^2)*sum(Tauik(:,k)) ...
                + (1+ delta^2)*sum(Tauik(:,k).*Dik(:,k).*E1ik(:,k)/sigmak) ...
                - delta* sum(Tauik(:,k).*(Wik(:,k).*(Dik(:,k).^2) + E2ik(:,k)/(sigmak^2))),[-1, 1]);%[-1, 1]
%  Deltak(k)= .01; 
   
            Lambdak(k) = Deltak(k)/sqrt(1 - Deltak(k)^2);
            %% CM-Step 4: update the nuk (the robustness parameter)
            
            nu0 = Nuk(k);
            %solution below works
%             Nuk(k) = fzero(@(nu) - psi((nu)/2) + log((nu)/2) + 1 + sum(Tauik(:,k).*(E3ik(:,k) - Wik(:,k)))/sum(Tauik(:,k)), [.1, 200], 1e-6);%nu0);%
            
            % probably fzero has been updated in the recent matlab versions
            Nuk(k) = fzero(@(nu) - psi((nu)/2) + log((nu)/2) + 1 + sum(Tauik(:,k).*(E3ik(:,k) - Wik(:,k)))/sum(Tauik(:,k)), [.1, 200]);%nu0);%
        end
        %%
        
        % observed-data log-likelihood
        loglik = sum(log_sum_piik_fik) + res.reg_irls;% + regEM;
        
        if verbose_EM,fprintf(1, 'ECM - STMoE  : Iteration : %d   Log-lik : %f \n ',  iter,loglik); end
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik = [stored_loglik, loglik];
    end% end of an EM loop
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    
    %%% results
    param.Alphak = Alphak;
    param.Betak = Betak;
    param.Sigmak = sqrt(Sigma2k);
    param.Lambdak = Lambdak;
    param.Nuk = Nuk;
        
    solution.param = param;
    solution.Deltak = Deltak;

    Piik = Piik(1:m,:);
    Tauik = Tauik(1:m,:);
    solution.Piik = Piik;
    solution.Tauik = Tauik;
    solution.log_piik_fik = log_piik_fik;
    solution.ml = loglik;
    solution.stored_loglik = stored_loglik;
    %% parameter vector of the estimated SNMoE model
    Psi = [param.Alphak(:); param.Betak(:); param.Sigmak(:); param.Lambdak(:); param.Nuk(:)];
    %
    solution.Psi = Psi;
    
    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
    [klas, Zik] = MAP(solution.Tauik);%solution.Piik);
    solution.klas = klas;
    
    % Statistics (means, variances)
    
    Xi_nuk = sqrt(Nuk/pi).*(gamma(Nuk/2 -1/2))./(gamma(Nuk/2));%((1- 3./(4*Nuk-1)).^(-1)).*sqrt(2./Nuk);%%%%
    % E[yi|zi=k]
    Ey_k = XBeta(1:m,:)*Betak + ones(m,1)*(Deltak.*sqrt(Sigma2k).*Xi_nuk );
    solution.Ey_k = Ey_k;
    % E[yi]
    Ey = sum(Piik.*Ey_k,2);
    solution.Ey = Ey;
    
    % Var[yi|zi=k]
    Vary_k = (Nuk./(Nuk-2) - (Deltak.^2).*(Xi_nuk.^2)).*Sigma2k;
    %ones(m,1)*((Nuk./(Nuk-2)).*Sigma2k) - Ey_k.^2;
    
    
    %(XBeta*Betak).^2 + 2*XBeta*Betak.*(ones(m,1)*(Deltak.*sqrt(Sigma2k).*Xi_nuk)) + ones(m,1)*((Nuk./(Nuk-2)).*Sigma2k);
    
    solution.Vary_k = Vary_k;
    
    % Var[yi]
    Vary = sum(Piik.*(Ey_k.^2 + ones(m,1)*Vary_k),2) - Ey.^2;
    solution.Vary = Vary;

    
    %%% BIC AIC et ICL
    lenPsi = length(Psi);
    solution.lenPsi = lenPsi;
    
    solution.BIC = solution.ml - (lenPsi*log(n*m)/2);
    solution.AIC = solution.ml - lenPsi;
    %% CL(theta) : complete-data loglikelihood
    zik_log_piik_fk = (repmat(Zik,n,1)).*solution.log_piik_fik;
    sum_zik_log_fik = sum(zik_log_piik_fk,2);
    comp_loglik = sum(sum_zik_log_fik);
    solution.CL = comp_loglik;
    solution.ICL = solution.CL - (lenPsi*log(n*m)/2);
    solution.XBeta = XBeta(1:m,:);
    solution.XAlpha = XAlpha(1:m,:);
    
    %%
    
    if total_EM_tries>1
        fprintf(1,'ml = %f \n',solution.ml);
    end
    if loglik > best_loglik
        best_solution = solution;
        best_loglik = loglik;
    end
end%fin de la premiï¿½re boucle while
solution = best_solution;
%
if total_EM_tries>1;   fprintf(1,'best loglik:  %f\n',solution.ml); end

solution.cputime = mean(stored_cputime);
solution.stored_cputime = stored_cputime;


