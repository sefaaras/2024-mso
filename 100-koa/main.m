paths;
algorithms = { 'koa_case1','koa_case2','koa_case3','koa_case4','koa_case5','koa_case6','koa_case7','koa_case8'}; % algorithm
dimension = 20; % (2, 10, 20)
maxFE = 10000; % 1000000

cec2022 = str2func('cec22_test_func');
globalMins = {300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700};
experimentNumber = 1; run = 21;
filename = 'result-';
functionsNumber = 12;
solution = zeros(experimentNumber, functionsNumber, run);
solutionR = zeros(functionsNumber * experimentNumber, run);

for ii = 1 : length(algorithms)
    disp(algorithms(ii));
    algorithm = algorithms{ii}; % Doğrudan fonksiyon adını al
    for i = 1 : functionsNumber
        disp(i);
        for j = 1 : run
            [~, bestFitness, ~] = feval(algorithm, cec2022, dimension, maxFE, i); % feval ile fonksiyonu çağır
            solution(1, i, j) = bestFitness - globalMins{i};
            for k = 1 : experimentNumber
                solutionR(k + experimentNumber * (i - 1), j) = solution(k, i, j);
            end
        end
    end     
    xlswrite(strcat(filename, algorithm, '-d=', num2str(dimension), '.xlsx'), solutionR, 1);
    eD = strcat(algorithm, '-Bitti :)');
    disp(eD);
end
