function [pbest,zbest,fval]=fit_RCs(data)
[pbest,zbest,fval] =fit_eistoolbox( ...
    data, ...           % input data
    's(R1,C1)', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [10,1e-6], ...          % initial parameters
    [0,0], ...             % lower boundary conditions for parameters
    [1000,1], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    10000 ...         % max number of iterations
    );

end