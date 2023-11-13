function [z_corrected,CPE_param] = correct_CPE_para(data,lfb,ufb)

pbest=fit_eistoolbox( ...
    data, ...           % input data
    's(R1,p(R1,E2))', ...  % circuit model (string representation)
    lfb, ...            % lower frequency bound
    ufb, ...            % upper frequency bound
    [3,10000,8e-6,0.7], ...          % initial parameters
    [0,100,0,0], ...             % lower boundary conditions for parameters
    [100,inf,1,1], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    10000 ...         % max number of iterations
    );
CPE_param=pbest;
z_corrected=data;
for i=1:size(data,1)-1
    z_CPE=computecircuit(pbest(i,2:end),'p(R1,E2)',data(1,:));
    z_CPE=z_CPE(:,1)+1j*z_CPE(:,2);
    a=-1;
    z_corrected(i+1,:)=(data(i+1,:).^a-z_CPE'.^a).^(1/a);
end
end