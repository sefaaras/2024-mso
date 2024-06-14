function [bestSolution, bestFitness, iteration]=koa(fhd, dimension, maxIteration, fNumber)

config;

SearchAgents_no=25; % Number of search agents (Planets)
Tmax=maxIteration; % Maximum number of Function evaluations
dim=dimension;
lb = lbArray;
ub = ubArray;

%%%%-------------------Definitions--------------------------%%
Sun_Pos=zeros(1,dim); % A vector to include the best-so-far Solution, representing the Sun
Sun_Score=inf; % A Scalar variable to include the best-so-far score
Convergence_curve=zeros(1,Tmax);

%%-------------------Controlling parameters--------------------------%%
Tc=3;
M0=0.1;
lambda=15;

%% Step 1: Initialization process
orbital=rand(1,SearchAgents_no); % Orbital Eccentricity (e) Eq.(4)
T=abs(randn(1,SearchAgents_no)); % Orbital Period (T) Eq.(5)
Positions=initialization(SearchAgents_no,dim,ub,lb); % Initialize the positions of planets
t=0; % Function evaluation counter 

%%---------------------Evaluation-----------------------%%
for i=1:SearchAgents_no
    PL_Fit(i)=testFunction(Positions(i,:)', fhd, fNumber); % Calculate fitness
    if PL_Fit(i)<Sun_Score % Update the best-so-far solution
       Sun_Score=PL_Fit(i);
       Sun_Pos=Positions(i,:);
    end
end

while t<Tmax
    [~, Order] = sort(PL_Fit);  % Sorting the fitness values
    worstFitness = PL_Fit(Order(SearchAgents_no)); % The worst Fitness value Eq.(11)
    M=M0*(exp(-lambda*(t/Tmax))); % Eq. (12)
    
    for i=1:SearchAgents_no
        R(i)=norm(Sun_Pos - Positions(i,:)); % Euclidean distance Eq.(7)
    end
    
    for i=1:SearchAgents_no
        sumFit = sum(PL_Fit - worstFitness);
        MS(i)=rand*(Sun_Score - worstFitness)/sumFit; % Mass of the Sun Eq.(8)
        m(i)=(PL_Fit(i) - worstFitness)/sumFit; % Mass of the planet Eq.(9)
    end
    
    for i=1:SearchAgents_no
        Rnorm(i)=(R(i) - min(R)) / (max(R) - min(R)); % Normalized R Eq.(24)
        MSnorm(i)=(MS(i) - min(MS)) / (max(MS) - min(MS)); % Normalized MS
        Mnorm(i)=(m(i) - min(m)) / (max(m) - min(m)); % Normalized m
        Fg(i)=orbital(i) * M * ((MSnorm(i) * Mnorm(i)) / (Rnorm(i)^2 + eps)) + rand; % Gravitational force Eq.(6)
    end
    
    for i=1:SearchAgents_no
        a1(i)=rand * (T(i)^2 * (M * (MS(i) + m(i)) / (4 * pi^2)))^(1/3); % Semi-major axis Eq.(23)
    end
    
    for i=1:SearchAgents_no
        a2=-1 + -1 * (mod(t, Tmax / Tc) / (Tmax / Tc)); % Cyclic control parameter Eq.(29)
        n=(a2 - 1) * rand + 1; % Linearly decreasing factor Eq.(28)
        a=randi(SearchAgents_no); % Random index
        b=randi(SearchAgents_no); % Random index
        rd=rand(1,dim); % Normally distributed vector
        r=rand; % Random number in [0,1]
        U1=rd < r; % Binary vector Eq.(21)
        O_P=Positions(i,:); % Store current position
        
        if rand < rand
            h=(1 / (exp(n .* randn))); % Adaptive factor Eq.(27)
            Xm=(Positions(b,:) + Sun_Pos + Positions(i,:)) / 3.0; % Average vector
            Positions(i,:)=Positions(i,:) .* U1 + (Xm + h * (Xm - Positions(a,:))) .* (1 - U1); % Update position Eq.(26)
        else
            if rand < 0.5
                f=1;
            else
                f=-1;
            end
            L=(M * (MS(i) + m(i)) * abs((2 / (R(i) + eps)) - (1 / (a1(i) + eps))))^(0.5); % Velocity Eq.(15)
            U=rd > rand(1,dim); % Binary vector
            if Rnorm(i) < 0.5
                M=(rand * (1 - r) + r); % Update mass Eq.(16)
                l=L * M * U; % Update velocity Eq.(14)
                Mv=(rand * (1 - rd) + rd); % Update mass velocity Eq.(20)
                l1=L * Mv * (1 - U); % Update velocity Eq.(19)
                V(i,:)=l .* (2 * rand * Positions(i,:) - Positions(a,:)) + l1 .* (Positions(b,:) - Positions(a,:)) + (1 - Rnorm(i)) * f * U1 .* rand(1,dim) .* (ub - lb); % Update velocity Eq.(13a)
            else
                U2=rand > rand; % Binary vector Eq. (22)
                V(i,:)=rand .* L .* (Positions(a,:) - Positions(i,:)) + (1 - Rnorm(i)) * f * U2 * rand(1,dim) .* (rand * ub - lb); % Update velocity Eq.(13b)
            end
            
            if rand < 0.5
                f=1;
            else
                f=-1;
            end
            Positions(i,:) = (Positions(i,:) + V(i,:) .* f) + (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i,:)); % Update position Eq.(25)
        end
        
        if rand < rand
            for j=1:dim
                if Positions(i,j) > ub(j)
                    Positions(i,j) = lb(j) + rand * (ub(j) - lb(j));
                elseif Positions(i,j) < lb(j)
                    Positions(i,j) = lb(j) + rand * (ub(j) - lb(j));
                end
            end
        else
            Positions(i,:) = min(max(Positions(i,:), lb), ub);
        end
        
        PL_Fit1 = testFunction(Positions(i,:)', fhd, fNumber); % Calculate fitness
        
        if PL_Fit1 < PL_Fit(i)
            PL_Fit(i) = PL_Fit1;
            if PL_Fit(i) < Sun_Score
                Sun_Score = PL_Fit(i);
                Sun_Pos = Positions(i,:);
            end
        else
            Positions(i,:) = O_P;
        end
        
        t = t + 1;
        if t > Tmax
            break;
        end
        
        Convergence_curve(t) = Sun_Score;
    end
    
    %% FDB rehberliğini burada kullanarak populasyonun çeşitliliğini arttırın
    index = fitnessDistanceBalance(Positions, PL_Fit);
    if PL_Fit(index) < Sun_Score
        Sun_Score = PL_Fit(index);
        Sun_Pos = Positions(index,:);
    end
end

bestSolution = Sun_Pos;
bestFitness = Sun_Score;
iteration = t;

end

function Positions = initialization(SearchAgents_no, dim, ub, lb)
Boundary_no = length(ub);
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
else
    for i = 1:dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:,i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;      
    end
end
end

function index = fitnessDistanceBalance(population, fitness)
[~, bestIndex] = min(fitness); 
best = population(bestIndex, :);
[populationSize, dimension] = size(population);

distances = zeros(1, populationSize); 
normFitness = zeros(1, populationSize); 
normDistances = zeros(1, populationSize); 
divDistances = zeros(1, populationSize); 

if min(fitness) == max(fitness)
    index = randi(populationSize);
else
    for i = 1 : populationSize
        value = 0;
        for j = 1 : dimension
            value = value + abs(best(j) - population(i, j));
        end
        distances(i) = value;
    end

    minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
    minDistance = min(distances); maxMinDistance = max(distances) - minDistance;

    for i = 1 : populationSize
        normFitness(i) = 1 - ((fitness(i) - minFitness) / maxMinFitness);
        normDistances(i) = (distances(i) - minDistance) / maxMinDistance;
        divDistances(i) = normFitness(i) + normDistances(i);
    end

    [~, index] = max(divDistances);
end
end
