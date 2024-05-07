% function [x,fval,ConvergenceCurve] = GEO (fun,nvars,lb,ub,options)

function[bestSolution, bestFitness, iteration] = GEO(cec20so, dimension, maxFes, i)

%% initialization
lb=ones(1, dimension) * -100;
ub=ones(1, dimension) * 100;
PopulationSize =50;
MaxIterations =maxFes;
fhd=cec20so;
fNumber=i;
nvars=dimension;
ConvergenceCurve  = zeros (1, MaxIterations);

x = lb + rand (PopulationSize,nvars) .* (ub-lb);

% FitnessScores = fun (x);
FitnessScores=testFunction(x', fhd, fNumber);


% solver-specific initialization
FlockMemoryF = FitnessScores;
FlockMemoryX = x;

options.AttackPropensity = [0.5 ,   2];
options.CruisePropensity = [1   , 0.5];
AttackPropensity = linspace (options.AttackPropensity(1), options.AttackPropensity(2), MaxIterations);
CruisePropensity = linspace (options.CruisePropensity(1), options.CruisePropensity(2), MaxIterations);

%% main loop
MaxIterations=MaxIterations/PopulationSize;
for CurrentIteration = 1 : MaxIterations
	
	% prey selection (one-to-one mapping)
	DestinationEagle = randperm (PopulationSize)';
	
	% calculate AttackVectorInitial (Eq. 1 in paper)
	AttackVectorInitial = FlockMemoryX (DestinationEagle,:) - x;
	
	% calculate Radius
	Radius = VecNorm (AttackVectorInitial, 2, 2);
	
	% determine converged and unconverged eagles
	ConvergedEagles = sum (Radius,2) == 0;
	UnconvergedEagles = ~ ConvergedEagles;
	
	% initialize CruiseVectorInitial
	CruiseVectorInitial = 2 .* rand (PopulationSize, nvars) - 1; % [-1,1]
	
	% correct vectors for converged eagles
	AttackVectorInitial (ConvergedEagles, :) = 0;
	CruiseVectorInitial (ConvergedEagles, :) = 0;
	
	% determine constrained and free variables
	for i1 = 1 : PopulationSize
		if UnconvergedEagles (i1)
			vConstrained = false ([1, nvars]); % mask
			idx = datasample (find(AttackVectorInitial(i1,:)), 1, 2);
			vConstrained (idx) = 1;
			vFree = ~vConstrained;
			CruiseVectorInitial (i1,idx) = - sum(AttackVectorInitial(i1,vFree).*CruiseVectorInitial(i1,vFree),2) ./ (AttackVectorInitial(i1,vConstrained)); % (Eq. 4 in paper)
		end
	end
	
	% calculate unit vectors
	AttackVectorUnit = AttackVectorInitial ./ VecNorm (AttackVectorInitial, 2, 2);
	CruiseVectorUnit = CruiseVectorInitial ./ VecNorm (CruiseVectorInitial, 2, 2);
	
	% correct vectors for converged eagles
	AttackVectorUnit(ConvergedEagles,:) = 0;
	CruiseVectorUnit(ConvergedEagles,:) = 0;
	
	% calculate movement vectors
	AttackVector = rand (PopulationSize, 1) .* AttackPropensity(CurrentIteration) .* Radius .* AttackVectorUnit; % (first term of Eq. 6 in paper)
	CruiseVector = rand (PopulationSize, 1) .* CruisePropensity(CurrentIteration) .* Radius .* CruiseVectorUnit; % (second term of Eq. 6 in paper)
	StepVector = AttackVector + CruiseVector;
	
	% calculate new x
	x = x + StepVector;
	
	% enforce bounds
	lbExtended = repmat (lb,[PopulationSize,1]);
	ubExtended = repmat (ub,[PopulationSize,1]);
	
	lbViolated = x < lbExtended;
	ubViolated = x > ubExtended;
	
	x (lbViolated) = lbExtended (lbViolated);
	x (ubViolated) = ubExtended (ubViolated);
	
	% calculate fitness
% 	FitnessScores = fun (x);
    FitnessScores=testFunction(x', fhd, fNumber);

	
	% update memory
	UpdateMask = FitnessScores < FlockMemoryF;
	FlockMemoryF (UpdateMask) = FitnessScores (UpdateMask);
	FlockMemoryX (UpdateMask,:) = x (UpdateMask,:);
	
	% update convergence curve
	ConvergenceCurve (CurrentIteration) = min (FlockMemoryF);
	
end
CurrentIteration=CurrentIteration+1;
%% return values

[fval, fvalIndex] = min (FlockMemoryF);
x = FlockMemoryX (fvalIndex, :);

bestSolution=x;
bestFitness=fval;
iteration=CurrentIteration;
end
