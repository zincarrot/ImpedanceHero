function [pbest,zbest,fval]=fit_Zres_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    's(p(R1,s(C1,L1)),p(R1,C1))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [10,1e-8,1e-5,180,5e-10], ...          % initial parameters
    [0,0,0,0,0], ...             % lower boundary conditions for parameters
    [50,1,1,500,1e-5], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );

end