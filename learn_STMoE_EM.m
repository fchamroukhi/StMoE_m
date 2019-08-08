function solution = learn_STMoE_EM(Y, x, K, p, q, ...
    total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)

% Faicel CHAMROUKHI
% (Mai 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off
if nargin<10, verbose_IRLS = 0; end
if nargin<9,  verbose_IRLS = 0; verbose_EM = 0; end
if nargin<8,  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; end
if nargin<7,  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; end
if nargin<6,  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; total_EM_tries=1;end

if size(Y,2)==1, Y=Y'; end % cas d'une courbe

[n, m] = size(Y); % n curves, each curve is composed of m observations

% construct the regression design matrices
XBeta = designmatrix_Poly_Reg(x,p); % for the polynomial regression
XAlpha = designmatrix_Poly_Reg(x,q); % for the logistic regression

XBeta  = repmat(XBeta,n,1);
XAlpha = repmat(XAlpha,n,1);


y = reshape(Y',[],1);

best_loglik = -inf;
stored_cputime = [];
EM_try = 1;
while (EM_try <= total_EM_tries)
    if total_EM_tries>1, fprintf(1, 'EM run n°  %d\n',EM_try); end
    time = cputime;
    %% EM Initialisation
    %1. Initialisation of Alphak's, Betak's and Sigmak's
    [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y, K, XAlpha, XBeta, 1);
    % 2. Intitialization of the skewness parameter
    Lambdak = -1 + 2*rand(1,K);
    Deltak  =  Lambdak./sqrt((1 + Lambdak.^2));
    % 3. Intitialization of the degrees of freedm
    Nuk = 50*rand(1,K);
    
    %     % or initialize it from an initial rough fit (only few iteration) for a TMoE (SNMoE or NMoE)
    %     model
    %     TMoE = learn_univ_TMoE_EM(Y, x, K, p, q, 1, 100, threshold, 0, 0);
    %     Alphak = TMoE.param.Alphak; Betak = TMoE.param.Betak; Sigma2k = (TMoE.param.Sigmak).^2; Nuk = TMoE.param.Nuk;
    %     Lambdak = -1 + 2*rand(1,K);
    %     Deltak  =  Lambdak./sqrt((1 + Lambdak.^2));
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
        
        % Allocate needed quantities for the E-Step
        Dik = zeros(m*n,K);
        Mik = zeros(m*n,K);
        % Allocate the E-Step quantities
        Wik = zeros(m*n,K);
        E1ik = zeros(m*n,K);
        E2ik = zeros(m*n,K);
        E3ik = zeros(m*n,K);
        % Allocate expert conditional density f(yi|xi, zi=k)
        fik = zeros(m*n,K);
        
        % Allocate possibly needed quantity for E3ik
        Integgtx = zeros(m*n,K);
        for k = 1:K
            muk = XBeta*Betak(:,k);
            sigmak = sqrt(Sigma2k(k));
            deltak = Deltak(k);
            
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            
            % Calculate the # expectations: W, Tau, E1, E2, E3: Equations 19--23
            
            % Wik: E[Wi|yi, xi, zi=k]
            Wik(:,k) = ((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2)).*...
                tcdf(Mik(:,k)*sqrt(1 + (2/(Nuk(k) + 1))), Nuk(k) + 3)./tcdf(Mik(:,k), Nuk(k) + 1);
            
            % value for STMoE density
            [STMoE, ~] = univ_STMoE_pdf(y, Alphak, Betak, sqrt(Sigma2k), Lambdak, Nuk, XBeta, XAlpha) ;
            
            % E1ik: E[Wi Ui |yi, xi, zi=k]
            E1ik(:,k) = deltak * (y - muk).*Wik(:,k) + ...
                (sqrt(1 - deltak^2)./(pi * STMoE)).*...
                ((Dik(:,k).^2/(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            
            % E2ik: E[Wi Ui^2|yi, xi, zi=k]
            E2ik(:,k) = deltak^2 * ((y - muk).^2).*Wik(:,k) + (1 - deltak^2)*sigmak^2 + ...
                ((deltak * (y - muk) * sqrt(1 - deltak^2))./(pi * STMoE)).*...
                (((Dik(:,k).^2)./(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            
            % E3ik: E[log(Wi)|yi,zi=k]
            fx = @(x) (psi((Nuk(k)+2)/2) - psi((Nuk(k)+1)/2) + log(1 + (x.^2)/(Nuk(k))) + ((Nuk(k)+1)*x.^2 - Nuk(k) - 1)./((Nuk(k)+1)*(Nuk(k)+1+x.^2))).*tpdf(x, Nuk(k) + 1);
            for i=1:n
                Integgtx(i,k) = integral(fx, -inf, Mik(i,k));
            end
            % >>> if OLS method, uncomment the above, and put Integgtx = 0;
            % (but the OLS doesnt guarantee the loglikelihood is increasing)
            E3ik(:,k) = Wik(:,k) - log((Nuk(k) + Dik(:,k).^2)/2) -(Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2) + psi((Nuk(k) + 1)/2)...
                + ((Lambdak(k)*Dik(:,k).*(Dik(:,k).^2 - 1))./sqrt((Nuk(k) + 1)*((Nuk(k) + Dik(:,k).^2).^3))).*...
                tpdf(Mik(:,k), Nuk(k)+1)./tcdf(Mik(:,k), Nuk(k) + 1) + (1./tcdf(Mik(:,k), Nuk(k) + 1)).*Integgtx(:,k);
            
            % skew t linear expert density
            fik(:,k) = (2/sigmak).*tpdf(Dik(:,k), Nuk(k)).*tcdf(Mik(:,k), Nuk(k) + 1);
        end
        
        % calculate the gating network probs
        Piik = multinomial_logit(Alphak, XAlpha);
        piik_fik = Piik.*fik;
        log_piik_fik = log(piik_fik);
        STMoE = sum(piik_fik,2); % skew-t mixture of experts density
        
        %E[Zi=k|yi, xi]
        Tauik = piik_fik./(STMoE*ones(1,K));
        
        %% CM-Step
        % updates of alphak's, betak's, sigma2k's and lambdak's
        % -----------------------------------------------------%
        %% CM-Step 1 update of the softmax parameters (Alphak)
        % IRLS for multinomial logistic regression
        res = IRLS(XAlpha, Tauik, Alphak,verbose_IRLS);
        Alphak = res.W;
        for k=1:K
            %% CM-Step 2: update the regression parameters (betak's and sigma2k's)
            %% update the regression coefficients betak's (eq. 26)
            %             TauikWik = (Tauik(:,k).*Wik(:,k))*ones(1,p+1);
            %             TauikX = XBeta.*(Tauik(:,k)*ones(1,p+1));
            %             betak =((TauikWik.*XBeta)'*XBeta)\(TauikX'*(Wik(:,k).*y - Deltak(k)*E1ik(:,k)));
            X= XBeta;
            tw = (Tauik(:,k).*Wik(:,k))*ones(1,p+1);%[n*(p+1)]
            yk = Tauik(:,k).*(Wik(:,k).*y - Deltak(k)*E1ik(:,k));%(nx1)
            %betak = inv((tw.*X)'*X)*X'*yk
            betak = (tw.*X)'*X\X'*yk;
            
            Betak(:,k) = betak;
            
            % recalculate E
            
            [STMoE, ~] = univ_STMoE_pdf(y, Alphak, Betak, sqrt(Sigma2k), Lambdak, Nuk, XBeta, XAlpha) ;
            muk = XBeta*Betak(:,k);
            sigmak = sqrt(Sigma2k(k));
            deltak = Deltak(k);
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            % Calculate the # expectations: W, Tau, E1, E2, E3: Equations 19--23
            % Wik: E[Wi|yi, xi, zi=k]
            Wik(:,k) = ((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2)).*...
                tcdf(Mik(:,k)*sqrt(1 + (2/(Nuk(k) + 1))), Nuk(k) + 3)./tcdf(Mik(:,k), Nuk(k) + 1);
            % E1ik: E[Wi Ui |yi, xi, zi=k]
            E1ik(:,k) = deltak * (y - muk).*Wik(:,k) + ...
                (sqrt(1 - deltak^2)./(pi * STMoE)).*...
                ((Dik(:,k).^2/(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            % E2ik: E[Wi Ui^2|yi, xi, zi=k]
            E2ik(:,k) = deltak^2 * ((y - muk).^2).*Wik(:,k) + (1 - deltak^2)*sigmak^2 + ...
                ((deltak * (y - muk) * sqrt(1 - deltak^2))./(pi * STMoE)).*...
                (((Dik(:,k).^2)./(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            %% update the variances sigma2k's (eq. 27)
            Sigma2k(k)= sum(Tauik(:,k).*(Wik(:,k).*((y-XBeta*betak).^2) - 2*Deltak(k)*E1ik(:,k).*(y-XBeta*betak)+ E2ik(:,k)))/...
                (2*(1-Deltak(k)^2)*sum(Tauik(:,k)));
            
            % recalculate E
            %    [STMoE, ~] = univ_STMoE_pdf(y, Alphak, Betak, sqrt(Sigma2k), Lambdak, Nuk, XBeta, XAlpha) ;
            muk = XBeta*Betak(:,k);
            sigmak = sqrt(Sigma2k(k));
            deltak = Deltak(k);
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            % Calculate the # expectations: W, Tau, E1, E2, E3: Equations 19--23
            % Wik: E[Wi|yi, xi, zi=k]
            Wik(:,k) = ((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2)).*...
                tcdf(Mik(:,k)*sqrt(1 + (2/(Nuk(k) + 1))), Nuk(k) + 3)./tcdf(Mik(:,k), Nuk(k) + 1);
            % E1ik: E[Wi Ui |yi, xi, zi=k]
            E1ik(:,k) = deltak * (y - muk).*Wik(:,k) + ...
                (sqrt(1 - deltak^2)./(pi * STMoE)).*...
                ((Dik(:,k).^2/(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            % E2ik: E[Wi Ui^2|yi, xi, zi=k]
            E2ik(:,k) = deltak^2 * ((y - muk).^2).*Wik(:,k) + (1 - deltak^2)*sigmak^2 + ...
                ((deltak * (y - muk) * sqrt(1 - deltak^2))./(pi * STMoE)).*...
                (((Dik(:,k).^2)./(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            %% CM-Step 3: update the skewness parameters lambdak's (eq. 28)
            lambda0 = Lambdak(k);
            try
                Lambdak(k) = fzero(@(lmbda) (lmbda/sqrt(1+lmbda^2))*(1-(lmbda^2/(1+lmbda^2)))*sum(Tauik(:,k)) ...
                    + (1+ (lmbda^2/(1+lmbda^2)))*sum(Tauik(:,k).*Dik(:,k).*E1ik(:,k)/sigmak) ...
                    - (lmbda/sqrt(1+lmbda^2))* sum(Tauik(:,k).*(Wik(:,k).*(Dik(:,k).^2) + E2ik(:,k)/(sigmak^2))), [-100, 100]);
            catch
                warning('The function in lambda doesnt differ in sign!');
                Lambdak(k) = lambda0;
            end
            % update the deltakak (the skewness parameter)
            Deltak(k) = Lambdak(k)/sqrt(1 + Lambdak(k)^2);
            
            
            %[STMoE, ~] = univ_STMoE_pdf(y, Alphak, Betak, sqrt(Sigma2k), Lambdak, Nuk, XBeta, XAlpha) ;
            sigmak = sqrt(Sigma2k(k));
            % calculate needed quantities for the E-Step
            Dik(:,k) = (y - muk)/sigmak;
            Mik(:,k) = Lambdak(k)*Dik(:,k).*sqrt((Nuk(k) + 1)./(Nuk(k) + (Dik(:,k)).^2));
            % Calculate the # expectations: W, Tau, E1, E2, E3: Equations 19--23
            %                 % Wik: E[Wi|yi, xi, zi=k]
            Wik(:,k) = ((Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2)).*...
                tcdf(Mik(:,k)*sqrt(1 + (2/(Nuk(k) + 1))), Nuk(k) + 3)./tcdf(Mik(:,k), Nuk(k) + 1);
            % E1ik: E[Wi Ui |yi, xi, zi=k]
            E1ik(:,k) = deltak * (y - muk).*Wik(:,k) + ...
                (sqrt(1 - deltak^2)./(pi * STMoE)).*...
                ((Dik(:,k).^2/(Nuk(k)*(1 - deltak^2)) + 1).^(-(Nuk(k)/2 + 1)));
            % E3ik: E[log(Wi)|yi,zi=k]
            fx = @(x) (psi((Nuk(k)+2)/2) - psi((Nuk(k)+1)/2) + log(1 + (x.^2)/(Nuk(k))) + ((Nuk(k)+1)*x.^2 - Nuk(k) - 1)./((Nuk(k)+1)*(Nuk(k)+1+x.^2))).*tpdf(x, Nuk(k) + 1);
            for i=1:n
                Integgtx(i,k) = integral(fx, -inf, Mik(i,k));
            end
            % >>> if OLS method, uncomment the above, and put Integgtx = 0;
            % (but the OLS doesnt guarantee the loglikelihood is increasing)
            E3ik(:,k) = Wik(:,k) - log((Nuk(k) + Dik(:,k).^2)/2) -(Nuk(k) + 1)./(Nuk(k) + Dik(:,k).^2) + psi((Nuk(k) + 1)/2)...
                + ((Lambdak(k)*Dik(:,k).*(Dik(:,k).^2 - 1))./sqrt((Nuk(k) + 1)*((Nuk(k) + Dik(:,k).^2).^3))).*...
                tpdf(Mik(:,k), Nuk(k)+1)./tcdf(Mik(:,k), Nuk(k) + 1) + (1./tcdf(Mik(:,k), Nuk(k) + 1)).*Integgtx(:,k);
            %% CM-Step 3: update the nuk's (the robustness parameters). eq. 29
            nu0 = Nuk(k);
            try
                Nuk(k) = fzero(@(nu) - psi(nu/2) + log(nu/2) + 1 + sum(Tauik(:,k).*(E3ik(:,k) - Wik(:,k)))/sum(Tauik(:,k)), [0, 100]);
            catch
                warning('The function in nu doesnt differ in sign!');
                Nuk(k) = nu0;
            end
        end
        %% observed-data log-likelihood
        loglik = sum(log(STMoE)) + res.reg_irls;
        
        if verbose_EM,fprintf(1, 'ECM-STMoE  : Iteration : %d   Log-lik : %f \n ',  iter,loglik); end
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik = [stored_loglik, loglik];
    end% end of an EM loop
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    
    % results
    param.Alphak = Alphak;
    param.Betak = Betak;
    param.Sigmak = sqrt(Sigma2k);
    param.Lambdak = Lambdak;
    param.Nuk = Nuk;
    
    solution.param = param;
    solution.Deltak = Deltak;
    
    Piik = Piik(1:m,:);
    Tauik = Tauik(1:m,:);
    solution.stats.Piik = Piik;
    solution.stats.Tauik = Tauik;
    solution.stats.log_piik_fik = log_piik_fik;
    solution.stats.ml = loglik;
    solution.stats.stored_loglik = stored_loglik;
    %% parameter vector of the estimated STMoE model
    Psi = [param.Alphak(:); param.Betak(:); param.Sigmak(:); param.Lambdak(:); param.Nuk(:)];
    %
    solution.stats.Psi = Psi;
    
    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
    [klas, Zik] = MAP(Tauik);
    solution.stats.klas = klas;
    
    % Statistics (means, variances)
    
    Xi_nuk = sqrt(Nuk/pi).*(gamma(Nuk/2 -1/2))./(gamma(Nuk/2));%((1- 3./(4*Nuk-1)).^(-1)).*sqrt(2./Nuk);%%%%
    % E[yi|zi=k]
    Ey_k = XBeta(1:m,:)*Betak + ones(m,1)*(Deltak.*sqrt(Sigma2k).*Xi_nuk );
    solution.stats.Ey_k = Ey_k;
    % E[yi]
    Ey = sum(Piik.*Ey_k,2);
    solution.stats.Ey = Ey;
    
    % Var[yi|zi=k]
    Vy_k = (Nuk./(Nuk-2) - (Deltak.^2).*(Xi_nuk.^2)).*Sigma2k;
    %ones(m,1)*((Nuk./(Nuk-2)).*Sigma2k) - Ey_k.^2;
    
    
    %(XBeta*Betak).^2 + 2*XBeta*Betak.*(ones(m,1)*(Deltak.*sqrt(Sigma2k).*Xi_nuk)) + ones(m,1)*((Nuk./(Nuk-2)).*Sigma2k);
    
    solution.stats.Vy_k = Vy_k;
    
    % Var[yi]
    Vy = sum(Piik.*(Ey_k.^2 + ones(m,1)*Vy_k),2) - Ey.^2;
    solution.stats.Vy = Vy;
    
    
    %%% BIC AIC ICL
    df = length(Psi);
    solution.stats.df = df;
    
    solution.stats.BIC = solution.stats.ml - (df*log(n*m)/2);
    solution.stats.AIC = solution.stats.ml - df;
    %% CL(theta) : complete-data loglikelihood
    zik_log_piik_fk = (repmat(Zik,n,1)).*log_piik_fik;
    sum_zik_log_fik = sum(zik_log_piik_fk,2);
    comp_loglik = sum(sum_zik_log_fik);
    solution.stats.CL = comp_loglik;
    solution.stats.ICL = comp_loglik - (df*log(n*m)/2);
    solution.stats.XBeta = XBeta(1:m,:);
    solution.stats.XAlpha = XAlpha(1:m,:);
    
    %%
    
    if total_EM_tries>1; fprintf(1,'ml = %f \n',solution.stats.ml);end
    if loglik > best_loglik
        best_solution = solution;
        best_loglik = loglik;
    end
end% End of all EM tries
solution = best_solution;
%
if total_EM_tries>1;   fprintf(1,'best loglik:  %f\n',solution.stats.ml); end

solution.stats.cputime = mean(stored_cputime);
solution.stats.stored_cputime = stored_cputime;
end

% NB: using function calls to (re-)calculate the E-Step outside can make the code clearer but it
% seems it slows down the algorithm (I've tested it)
