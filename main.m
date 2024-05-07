paths;
cec2022 = str2func('cec22_test_func');

algorithms = {'sos'};
experimentNumber = 1; run = 1; dimension = 20;
maxFE = 100 * dimension; filename = 'result-';
functionsNumber = 12;
solution = zeros(experimentNumber, functionsNumber, run);
solutionR = zeros(functionsNumber * experimentNumber, run);

for ii = 1 : length(algorithms)
    disp(algorithms(ii));
    algorithm = str2func(char(algorithms(ii)));
    for i = 1 : functionsNumber
        disp(i);
        for j = 1 : run
            [~, bestFitness, ~] = algorithm(cec2022, dimension, maxFE, i);
            solution(1, i, j) = bestFitness;
            for k = 1 : experimentNumber
                solutionR(k + experimentNumber * (i - 1), j) = solution(k, i, j);
            end
        end
    end
%     xlswrite(strcat(filename, func2str(algorithm), '-d=', num2str(dimension), '.xlsx'), solutionR, 1);
    eD = strcat(func2str(algorithm), '-Bitti :)');
    disp(eD);
end