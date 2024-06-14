function index = tournamentSelection(Positions, PL_Fit, tournamentSize)
    % Rastgele turnuva boyutunda bireyler seç
    tournamentIndices = randperm(length(PL_Fit), tournamentSize);
    % Turnuva içindeki en iyi bireyi seç
    [~, bestIndex] = min(PL_Fit(tournamentIndices));
    index = tournamentIndices(bestIndex);
end