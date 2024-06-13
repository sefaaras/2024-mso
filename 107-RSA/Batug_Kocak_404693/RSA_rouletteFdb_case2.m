function [bestSolution, bestFitness, iteration]=RSA_rouletteFdb_case2(fhd, dimension, maxIteration, fNumber)
    config;
    LB = lbArray;
    UB = ubArray;
    Dim = dimension;
    T = maxIteration;
    N = 10;
    Best_P = zeros(1, Dim);           % best positions
    Best_F = inf;                     % best fitness
    X = initialization(N, Dim, UB, LB); %Initialize the positions of solution
    Xnew = zeros(N, Dim);
    Conv = zeros(1, T);               % Convergance array

    Alpha = 0.1;                      % the best value 0.1
    Beta = 0.005;                     % the best value 0.005
    Ffun = zeros(1, size(X, 1));      % (old fitness values)
    Ffun_new = zeros(1, size(X, 1));  % (new fitness values)

    for i = 1:size(X, 1)
        Ffun(1, i) = testFunction(X(i, :)', fhd, fNumber);
        if Ffun(1, i) < Best_F
            Best_F = Ffun(1, i);
            Best_P = X(i, :);
        end
    end

    Maxit = T + 1;
    Maxit = Maxit / N;
    for t = 1:Maxit  % Main loop %Update the Position of solutions
        ES = 2 * randn * (1 - (t / T));  % Probability Ratio

        % Calculate Fitness for all solutions in X
        for i = 1:size(X, 1)
            Ffun(1, i) = testFunction(X(i, :)', fhd, fNumber);
        end

        for i = 2:size(X, 1)
            % Randomly choose between FDB and random selection
            if rand < 0.5 
                selected_index = rouletteFitnessDistanceBalance(X, Ffun); 
            else
                selected_index = randi([1 size(X, 1)]);
            end

            for j = 1:size(X, 2)
                % Use the selected guide (Best_P from fitnessDistanceBalance)
                R = X(selected_index, j) - X(i, j) / (X(selected_index, j) + eps);
                P = Alpha + (X(i, j) - mean(X(i, :))) / (X(selected_index, j) * (UB(j) - LB(j)) + eps);
                Eta = X(selected_index, j) * P;

                if (t < T / 4)
                    Xnew(i, j) = X(selected_index, j) - Eta * Beta - R * rand;
                elseif (t < 2 * T / 4 && t >= T / 4)
                    Xnew(i, j) = X(selected_index, j) * X(i, j) * ES * rand;
                elseif (t < 3 * T / 4 && t >= 2 * T / 4)
                    Xnew(i, j) = X(selected_index, j) * P * rand;
                else
                    Xnew(i, j) = X(selected_index, j) - Eta * eps - R * rand;
                end
            end

            Flag_UB = Xnew(i, :) > UB; % check if they exceed (up) the boundaries
            Flag_LB = Xnew(i, :) < LB; % check if they exceed (down) the boundaries
            Xnew(i, :) = (Xnew(i, :) .* ~(Flag_UB + Flag_LB)) + UB .* Flag_UB + LB .* Flag_LB;
            Ffun_new(1, i) = testFunction(Xnew(i, :)', fhd, fNumber);
            if Ffun_new(1, i) < Ffun(1, i)
                X(i, :) = Xnew(i, :);
                Ffun(1, i) = Ffun_new(1, i);
            end
            if Ffun(1, i) < Best_F
                Best_F = Ffun(1, i);
                Best_P = X(i, :);
            end
        end

        Conv(t) = Best_F;  % Update the convergence curve
        if mod(t, 50) == 0  % Print the best universe details after every 50 iterations
            % display(['At iteration ', num2str(t), ' the best solution fitness is ', num2str(Best_F)]);
        end
    end
    t = t + 1;
    bestSolution = Best_P;
    bestFitness = Best_F;
    iteration = t;
end
