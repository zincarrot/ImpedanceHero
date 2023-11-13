function [pbest,zbest,fval] = fit_CPE_randles(data)

[pbest,zbest,fval] =fit_eistoolbox( ...
    data, ...           % input data
    's(R1,p(R1,E2))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [3,inf,8e-6,0.7], ...          % initial parameters
    [0,100,0,0], ...             % lower boundary conditions for parameters
    [100,inf,1,1], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    10000 ...         % max number of iterations
    );

end