function [bestSolution, bestFitness, iteration]=cmaes(fhd, dimension, maxIteration, fNumber)

config;
nVar = dimension;                % Number of Unknown (Decision) Variables
VarSize = [1 nVar];       % Decision Variables Matrix Size
VarMin = lbArray;             % Lower Bound of Decision Variables
VarMax = ubArray;             % Upper Bound of Decision Variables

%% CMA-ES Settings

% Maximum Number of Iterations
MaxIt = ceil(maxIteration / (121 + dimension/1.8));

% Population Size (and Number of Offsprings)
lambda = (4+round(3*log(nVar)))*10;

% Number of Parents
mu = round(lambda/2);

% Parent Weights
w = log(mu+0.5)-log(1:mu);
w = w/sum(w);

% Number of Effective Solutions
mu_eff = 1/sum(w.^2);

% Step Size Control Parameters (c_sigma and d_sigma);
sigma0 = 0.3*(VarMax-VarMin);
cs = (mu_eff+2)/(nVar+mu_eff+5);
ds = 1+cs+2*max(sqrt((mu_eff-1)/(nVar+1))-1, 0);
ENN = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));

% Covariance Update Parameters
cc = (4+mu_eff/nVar)/(4+nVar+2*mu_eff/nVar);
c1 = 2/((nVar+1.3)^2+mu_eff);
alpha_mu = 2;
cmu = min(1-c1, alpha_mu*(mu_eff-2+1/mu_eff)/((nVar+2)^2+alpha_mu*mu_eff/2));
hth = (1.4+2/(nVar+1))*ENN;

%% Initialization

ps = cell(MaxIt, 1);
pc = cell(MaxIt, 1);
C = cell(MaxIt, 1);
sigma = cell(MaxIt, 1);

ps{1} = zeros(VarSize);
pc{1} = zeros(VarSize);
C{1} = eye(nVar);
sigma{1} = sigma0;

empty_individual.Position = [];
empty_individual.Step = [];
empty_individual.Cost = [];

M = repmat(empty_individual, MaxIt, 1);
M(1).Position = unifrnd(VarMin, VarMax, VarSize);
M(1).Step = zeros(VarSize);
M(1).Cost = testFunction(M(1).Position', fhd, fNumber);

BestSol = M(1);

BestCost = zeros(MaxIt, 1);

%% CMA-ES Main Loop

for g = 1:MaxIt
    
    % Generate Samples
    pop = repmat(empty_individual, lambda, 1);
    for i = 1:lambda

        % Generating Sample
        pop(i).Step = mvnrnd(zeros(VarSize), C{g});
        pop(i).Position = M(g).Position + sigma{g} .* pop(i).Step;
        
        % Applying Bounds
        pop(i).Position = max(pop(i).Position, VarMin);
        pop(i).Position = min(pop(i).Position, VarMax);
        
        % Evaluation
        pop(i).Cost = testFunction(pop(i).Position', fhd, fNumber);
        
        % Update Best Solution Ever Found
        if pop(i).Cost<BestSol.Cost
            BestSol = pop(i);
        end
    end
    
    % Sort Population
    Costs = [pop.Cost];
    [~, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
  
    % Save Results
    BestCost(g) = BestSol.Cost;
        
    % Exit At Last Iteration
    if g == MaxIt
        break;
    end
    
    % Update Mean
    M(g+1).Step = 0;
    for j = 1:mu
        M(g+1).Step = M(g+1).Step+w(j)*pop(j).Step;
    end
    M(g+1).Position = M(g).Position + sigma{g} .* M(g+1).Step;
    
    % Applying Bounds
    M(g+1).Position = max(M(g+1).Position, VarMin);
    M(g+1).Position = min(M(g+1).Position, VarMax);
    
    % Evaluation
    M(g+1).Cost = testFunction(M(g+1).Position', fhd, fNumber);
    
    % Update Best Solution Ever Found
    if M(g+1).Cost < BestSol.Cost
        BestSol = M(g+1);
    end
    
    % Update Step Size
    ps{g+1} = (1-cs)*ps{g}+sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
    sigma{g+1} = sigma{g}*exp(cs/ds*(norm(ps{g+1})/ENN-1))^0.3;
    
    % Update Covariance Matrix
    if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1)))<hth
        hs = 1;
    else
        hs = 0;
    end
    delta = (1-hs)*cc*(2-cc);
    pc{g+1} = (1-cc)*pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
    C{g+1} = (1-c1-cmu)*C{g}+c1*(pc{g+1}'*pc{g+1}+delta*C{g});
    for j = 1:mu
        C{g+1} = C{g+1}+cmu*w(j)*pop(j).Step'*pop(j).Step;
    end
    
    % If Covariance Matrix is not Positive Defenite or Near Singular
    [V, E] = eig(C{g+1});
    if any(diag(E)<0)
        E = max(E, 0);
        C{g+1} = V*E/V;
    end
    
end

bestSolution=BestSol.Position;
bestFitness=BestSol.Cost;
iteration=g;

end