function solution = learn_univ_SNMoE_EM(Y, x, K, p, dim_w, total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learn_univ_SNMoE_EM: fits an SNMoE model with the Conditional EM algorithm
%
% Please cite the following papers for this code:
%
% @InProceedings{Chamroukhi-SNMoE-IJCNN-2016,
%     Author         = {F. Chamroukhi},
%     booktitle  = {The International Joint Conference on Neural Networks (IJCNN)},
%     Address = {Vancouver, Canada},
%     Title          = {Skew-Normal Mixture of Experts},
%     Year           = {2016},
% 	Month = {July},
% 	url = {https://chamroukhi.com/papers/Chamroukhi-SNMoE-IJCNN2016.pdf},
% 	slides = {./conf-presentations/FChamroukhi-IJCNN-2016-Talk.pdf},
% 	software =  {https://github.com/fchamroukhi/SNMoE_Matlab}
% 	}
%
% @article{Chamroukhi-NNMoE-2015,
% 	Author = {F. Chamroukhi},
% 	eprint = {arXiv:1506.06707},
% 	Title = {Non-Normal Mixtures of Experts},
% 	Volume = {},
% 	url= {http://arxiv.org/pdf/1506.06707.pdf},
% 	month = {July},
% 	Year = {2015},
% 	note = {Report (61 pages)}
% }
%
% @article{NguyenChamroukhi-MoE,
% 	Author = {Hien D. Nguyen and Faicel Chamroukhi},
% 	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
% 	Title = {Practical and theoretical aspects of mixture-of-experts modeling: An overview},
% publisher = {Wiley Periodicals, Inc},
% issn = {1942-4795},
% doi = {10.1002/widm.1246},
% pages = {e1246--n/a},
% keywords = {classification, clustering, mixture models, mixture of experts, neural networks},
% 	Month = {Feb},
% Year = {2018},
% url = {https://chamroukhi.com/papers/Nguyen-Chamroukhi-MoE-DMKD-2018}
% }
%
% Developed and written by Faicel Chamroukhi
% (c) F. Chamroukhi (2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off


if nargin<10, verbose_IRLS = 0; end
if nargin<9,  verbose_IRLS =0; verbose_EM = 0; end
if nargin<8,  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; end
if nargin<7,  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; end
if nargin<6,  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; total_EM_tries=1;end

if size(Y,2)==1, Y=Y'; end %
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
while EM_try <= total_EM_tries
    if total_EM_tries>1, fprintf(1, 'ECM run n°  %d  \n ', EM_try); end
    time = cputime;
    %% EM Initialisation
    
    %%1. Initialisation of Alphak's, Betak's and Sigmak's
    segmental = 1;
    [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y, K, XAlpha, XBeta, segmental);
    if EM_try ==1, Alphak = zeros(q+1,K-1);end % set the first initialization to the null vector
    %% initialize using moments
    %     ybar = mean(y);
    %     a1 = sqrt(2/pi);
    %     b1 = (4/pi - 1)*a1;
    %     m2 = (1/(m-1))*sum((y- ybar).^2);
    %     m3 = (1/(m-1))*sum(abs(y - ybar).^3);
    %     DeltakAll = (a1^2 + m2*(b1/m3)^(2/3))^(0.5);
    %     Deltak = DeltakAll*ones(1,K);
    
    % %% or initialize with NMoE
    %     solution = learn_univ_NMoE_EM(Y, x, K, p, q, 1, 500, 1e-6, 1, 0);
    % %   total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)
    %     Alphak=solution.param.Alphak;
    %     Betak =solution.param.Betak;
    %     Sigmak =sqrt(solution.param.Sigmak);
    %     Sigma2k = Sigmak.^2;
    
    %% Initialize the skewness parameter Lambdak (by equivalence Deltak)
    Lambdak = -1 + 2*rand(1,K);
    Deltak = Lambdak./sqrt(1+Lambdak.^2);

    %%%
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    %% EM %%%%
    while ~converge && (iter< max_iter_EM)
        iter=iter+1;
        %% E-Step
        Piik = multinomial_logit(Alphak,XAlpha);
        
        piik_fik = zeros(m*n,K);
        
        %Dik = zeros(m*n,K);
        E1ik = zeros(m*n,K);
        E2ik = zeros(m*n,K);
        for k = 1:K
            
            muk = XBeta*Betak(:,k);
            sigma2k = Sigma2k(k);
            sigmak = sqrt(sigma2k);
            dik = (y - muk)/sigmak;
            %Dik(:,k) = dik;
            
            % E[Ui|yi,xi,zik=1] and E[Ui^2|yi,xi,zik=1]
            mu_uk = Deltak(k)*(y - muk);
            sigma2_uk = (1-Deltak(k)^2)*Sigma2k(k);
            sigma_uk = sqrt(sigma2_uk);
            
            % E1ik = E[Ui|yi,xi,zik=1] and E2ik
            E1ik(:,k) = mu_uk + sigma_uk * normpdf(Lambdak(k)*dik)./normcdf(Lambdak(k)*dik);
            % E2ik = E[Ui^2|y,zik=1]
            E2ik(:,k) = mu_uk.^2 + sigma_uk.^2 + sigma_uk*mu_uk.*normpdf(Lambdak(k)*dik)./normcdf(Lambdak(k)*dik);
            
            % piik*SN(.;muk;sigma2k;lambdak)
            %weighted skew normal linear expert likelihood
            piik_fik(:,k) = Piik(:,k).*(2/sigmak).*normpdf(dik).*normcdf(Lambdak(k)*dik);
        end
        log_piik_fik = log(piik_fik);
        log_sum_piik_fik = log(sum(piik_fik,2));
        
        % E[Zik|y,x] and E[U^2|y,zik=1]
        Tauik = piik_fik./(sum(piik_fik,2)*ones(1,K));
        
        %log_Tauik = log_piik_fik - logsumexp(log_piik_fik, 2)*ones(1,K);
        %Tauik = exp(log_Tauik);
        
        %% M-Step
        % updates of Alpha, betak's, sigma2k's and lambdak's
        % --------------------------------------------------%
        % update of the softmax parameters (Alphak)
        %  IRLS for multinomial logistic regression
        res = IRLS(XAlpha, Tauik, Alphak,verbose_IRLS);
        Piik = res.piik;
        Alphak = res.W;
        %%
        for k=1:K
            %%update the regression coefficients
            %             XBetak = XBeta.*(sqrt(Tauik(:,k))*ones(1,p+1));%[m*(p+1)]
            %             yk = sqrt(Tauik(:,k)).*y;% dimension :(nx1).*(nx1) = (nx1)
            %             %
            %             betak = (XBetak'*XBetak)\(XBetak'*(yk - (Deltak(k)./sqrt(Tauik(:,k))).*E1ik(:,k)));
            
            tauik_Xbeta = (Tauik(:,k)*ones(1,p+1)).*XBeta;
            betak = (tauik_Xbeta'*XBeta)\(tauik_Xbeta'*(y - Deltak(k)*E1ik(:,k)));
            
            Betak(:,k) = betak;
                        
            % % update the variances sigma2k
            Sigma2k(k)= sum(Tauik(:,k).*((y-XBeta*betak).^2 - 2*Deltak(k)*E1ik(:,k).*(y-XBeta*betak) + E2ik(:,k)))/...
                (2*(1-Deltak(k)^2)*sum(Tauik(:,k)));
            
            % update the lambdak (the skewness parameter)
            lambda0 = Lambdak(k);

            try
                Lambdak(k) = fzero(@(lmbda) Sigma2k(k)*(lmbda./sqrt(1+lmbda.^2))*(1-(lmbda.^2/(1+lmbda.^2)))*sum(Tauik(:,k)) ...
                    + (1+ (lmbda.^2/(1+lmbda.^2)))*sum(Tauik(:,k).*(y-XBeta*betak).*E1ik(:,k)) ...
                    - (lmbda./sqrt(1+lmbda.^2))* sum(Tauik(:,k).*(E2ik(:,k) + (y-XBeta*betak).^2)), lambda0);
            catch
                Lambdak(k) = lambda0;
            end

            % update the deltakak (the skewness parameter)
            Deltak(k) = Lambdak(k)/sqrt(1 + Lambdak(k)^2);
        end
        %%
        
        % observed-data log-likelihood
        loglik = sum(log_sum_piik_fik) + res.reg_irls;% + regEM;
        
        if verbose_EM,fprintf(1, 'ECM - SNMoE   : Iteration : %d   Log-lik : %f \n ',  iter,loglik); end
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik = [stored_loglik, loglik];
    end% end of an EM loop
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    
    %%% results
    param.Alphak = Alphak;
    param.Betak = Betak;
    Sigmak = sqrt(Sigma2k);
    param.Sigmak = Sigmak;
    param.Lambdak = Lambdak;
    
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
    Psi = [param.Alphak(:); param.Betak(:); param.Sigmak(:); param.Lambdak(:)];
    %
    solution.Psi = Psi;
    
    %Bayes' allocation rule to calculate a partition of the data
    [klas, Zik] = MAP(solution.Tauik);%solution.param.Piik);
    solution.klas = klas;
    
    % Statistics (mean and variances)
    
    % E[yi|xi,zi=k]
    Ey_k = XBeta(1:m,:)*Betak + ones(m,1)*( sqrt(2/pi)*Deltak.*Sigmak );
    solution.Ey_k = Ey_k;
    % E[yi|xi]
    Ey = sum(Piik.*Ey_k,2);
    solution.Ey = Ey;
    
    % Var[yi|xi,zi=k]
    Var_yk = (1 - (2/pi)*(Deltak.^2)).*(Sigmak.^2);
    solution.Vary_k = Var_yk;
    
    % Var[yi|xi]
    Var_y = sum(Piik.*(Ey_k.^2 + ones(m,1)*Var_yk),2) - Ey.^2;
    solution.Vary = Var_y;
    
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
end
solution = best_solution;
%
if total_EM_tries>1;   fprintf(1,'best loglik:  %f\n',solution.ml); end

solution.cputime = mean(stored_cputime);
solution.stored_cputime = stored_cputime;