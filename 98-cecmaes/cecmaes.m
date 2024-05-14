function [bestSolution, bestFitness, iteration]=cecmaes(fhd, dimension, maxIteration, fNumber)

config;

CE_CMAES_parameters.I_NP=100;
CE_CMAES_parameters.I_itermax=ceil(maxIteration/CE_CMAES_parameters.I_NP)+1;
%% Cross Entropy Method 
Xmin=lbArray;
Xmax=ubArray;
I_itermax=CE_CMAES_parameters.I_itermax;
fitMaxVector = nan(1,I_itermax);
Ne=round(0.2*CE_CMAES_parameters.I_NP);
D=dimension; % D=3408
% ccrand=rand(1,D);
% ccpos=((1./ccrand)-floor(1./ccrand));
%mu=Xmin+rand()*(Xmax-Xmin).*ccpos;
%mu=Xmin+(Xmax-Xmin).*ccpos; %1*D
mu=(Xmin+Xmax)/2; 
sigma2=(Xmax-Xmin)/4; %CE std deviation
alpha=0.9;
beta=0.1;
q=5;
xmin=repmat(Xmin, CE_CMAES_parameters.I_NP , 1);
xmax=repmat(Xmax, CE_CMAES_parameters.I_NP , 1);


gen=1;
while gen< round(0.5*I_itermax)
    for ii=1:CE_CMAES_parameters.I_NP
        pos(ii,:) = sigma2.*randn(1,D)+mu;
    end
    
    % check limits of x.(min and max limits)
    changemax=pos>xmax;
    pos(changemax)=xmax(changemax);
    changemin=pos<xmin;
    pos(changemin)=xmin(changemin);

solFitness_M = testFunction(pos', fhd, fNumber);

   mu_old=mu;
   sigma2_old=sigma2;
    [yy, I]= sort(solFitness_M, 'ascend');%  yy:1*particles, I:1*particles Best(f) to worst(f)(MIn to max fitness)
    pos=pos(I,:);
    gbestval=yy(1); %1*1
    fitMaxVector(1,gen) = gbestval;
    gbest=pos(1, :); % gbest x value 1*D
    %%%%%%%%%%%%%%%%
    xx=pos(1:Ne, :); %Elite particles Ne*D
    mu=mean(xx);     % Consider top best 10 x, columnwise mean of x1 and x2, 1*D
    sigma2=sqrt(var(xx,1)); % S.D., same as above column wise var of x1 and x2, 1*D
    
    mu=alpha*mu+(1-alpha)*mu_old; % new mu=(0.9*mu of elite particles)+(0.1* old mu of all particles), size 1*D
    sigmaStd1 = beta - beta * ( 1 - 1 / ( gen + 1 ) )^q;
    sigma2 = sigma2 * sigmaStd1 + sigma2_old * ( 1 - sigmaStd1 );
   % sigma2=beta*sigma2+(1-beta)*sigma2_old;% New S.D=(0.1*S.D of elite particles)+(0.9*old S.D of all particles), size 1*D
     
    gen=gen+1;
    fitMaxVector(1,gen) = gbestval;
  
 %%%%%%%%%%%%%%%%%%%
end % end of while of CE method
%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%
%% Apply CMAES method for local exploitation 
FVr_bestmemit=gbest;%1*D
xmean=FVr_bestmemit'; % gbest, D*1
sigma = 1.0;
lambda=CE_CMAES_parameters.I_NP; % no. of particles
mu = lambda/2;               % number of parents/points for recombination
 weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination, (particles/2)*1
 mu = floor(mu);        
 weights = weights/sum(weights);     % normalize recombination weights array, particles/2)*1
 mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i, 1*1
 
  % Strategy parameter setting: Adaptation
  cc = (4+mueff/D) / (D+4 + 2*mueff/D);  % time constant for cumulation for C, 1*1
  cs = (mueff+2) / (D+mueff+5);  % t-const for cumulation for sigma control, 1*1
  c1 = 2 / ((D+1.3)^2+mueff);    % learning rate for rank-one update of C, 1*1
  cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((D+2)^2+mueff));  % and for rank-mu update, 1*1
  damps = 1 + 2*max(0, sqrt((mueff-1)/(D+1))-1) + cs; % damping for sigma 
                                                      % usually close to 1,
                                                      % 1*1
                                                      
   % Initialize dynamic (internal) strategy parameters and constants
  pc = zeros(D,1); psc = zeros(D,1);   % evolution paths for C and sigma, D*1
  B = eye(D,D);                       % B defines the coordinate system diagonal 1 1 1
  F = ones(D,1);                      % diagonal D defines the scaling D *1 : 1 1 
  C = B * diag(F.^2) * B';            % covariance matrix C, D*D
  invsqrtC = B * diag(F.^-1) * B';    % C^-1/2, D*D
  eigeneval = 0;                      % track update of B and D
  chiN=D^0.5*(1-1/(4*D)+1/(21*D^2));  % expectation of 
                                      %   ||N(0,I)|| == norm(randn(N,1)),1*1 
   counteval = 0;                                   
                                                      

while gen>= round(0.5*I_itermax)&& gen<I_itermax 
    
    for k=1:lambda  %pop size
          
          arx(:,k) = xmean + sigma * B * (F .* randn(D,1)); % m + sig * Normal(0,C), arx=35*100, D*N
          %arx(:,k) = xmean + sigma * B * (F .* ccpos); % m + sig * Normal(0,C), arx=35*100
          counteval = counteval+1; %=N
    end
      arx_trans1=arx'; % 100*35, N*D
      
      
      % check limits of x.(min and max limits)
    changemax=arx_trans1>xmax;
    arx_trans1(changemax)=xmax(changemax);
    changemin=arx_trans1<xmin;
   arx_trans1(changemin)=xmin(changemin);
   %%%%%%%%%%%%%%%%%%%%%%
   arx_trans=arx_trans1; %size proc.pop_size*D, N*D
   pos=arx_trans;
   solFitness_M=testFunction(pos', fhd, fNumber);
   % Sort by fitness and compute weighted mean into xmean
      [~, arindex] = sort(solFitness_M); % minimization
      %%%%%%%%%%%%%%%%%%%
       % UPDATE GLOBAL BEST
    [ tmpgbestval, gbestid ] = min(solFitness_M);
    if tmpgbestval < gbestval
        gbestval = tmpgbestval;
        gbest =arx_trans ( gbestid, : );
    end
    
    fitMaxVector(1,gen) = gbestval;
    FVr_bestmemit=gbest; %gbest x value 1*D
      xold = xmean; %D*1
      xmean = arx(:,arindex(1:mu))*weights;   % recombination, new mean value, D*1
      
      % Cumulation: Update evolution paths
      psc = (1-cs)*psc ... 
            + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;% D*1 
      hsig = norm(psc)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4 + 2/(D+1);% counteval=N, logical
      pc = (1-cc)*pc ...
            + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma; %D*1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Adapt covariance matrix C
      artmp = (1/sigma) * (arx(:,arindex(1:mu))-repmat(xold,1,mu)); %D*50(mu)
      C = (1-c1-cmu) * C ...                  % regard old matrix  C=D*D
           + c1 * (pc*pc' ...                 % plus rank one update
                   + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
           + cmu * artmp * diag(weights) * artmp'; % plus rank mu update
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Adapt step size sigma
      sigma = sigma * exp((cs/damps)*(norm(psc)/chiN - 1)); %1*1 scalar
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Decomposition of C into B*diag(D.^2)*B' (diagonalization)
      if counteval - eigeneval > lambda/(c1+cmu)/D/10  % to achieve O(N^2)
          eigeneval = counteval;
          C = triu(C) + triu(C,1)'; % enforce symmetry, D*D
          [B,F] = eig(C);           % eigen decomposition, B==normalized eigenvectors, D*D
          F = sqrt(diag(F));        % D is a vector of standard deviations now D*1
          invsqrtC = B * diag(F.^-1) * B'; %D*D
      end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    gen=gen+1;
    fitMaxVector(1,gen) = gbestval;
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end % end of while gen...
Fit_and_p=[fitMaxVector(1,gen) 0]; %;p2;p3;p4]
bestSolution=FVr_bestmemit;
bestFitness=Fit_and_p(1);
iteration=1;
end
