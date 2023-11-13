function [z_corrected,z_CPE,CPE_param] = correct_CPE(data,lfb,ufb)

pbest=fit_eistoolbox( ...
    data, ...           % input data
    's(R1,E2)', ...  % circuit model (string representation)
    lfb, ...            % lower frequency bound
    ufb, ...            % upper frequency bound
    [3,8e-6,0.7], ...          % initial parameters
    [0,0,0], ...             % lower boundary conditions for parameters
    [2000,1,1], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    1000 ...         % max number of iterations
    );
CPE_param=pbest;
z_corrected=data;
z_CPE=data;
for i=1:size(data,1)-1
    z_CPE1=computecircuit(pbest(i,2:end),'E2',data(1,:));
    z_CPE1=z_CPE1(:,1)-1j*z_CPE1(:,2);
    z_CPE(i+1,:)=z_CPE1';
    z_corrected(i+1,:)=data(i+1,:)-z_CPE1';
end
end