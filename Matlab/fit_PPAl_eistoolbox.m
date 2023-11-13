function [Z_sig,Z_res,pbest,zbest,fval]=fit_PPAl_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    's(p(E2,R1),R1,p(R1,C1,s(R1,C1)))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [1e-5,0.7,1000,15,15,5e-7,2,1e-5], ...          % initial parameters
    [0,0.5,10,0,0,0,0,0], ...             % lower boundary conditions for parameters
    [2e-4,1,inf,30,30,1e-4,20,1e-3], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    3, ...      % weighting type number
    2000 ...         % max number of iterations
    );

f=data(1,:);
r=pbest(:,3);
n=pbest(:,2);
Q=pbest(:,1);
% C_s=pbest(:,1);

Z_sig=data;
Z_res=data;

Z_cpe=(1./((Q.*(2j*pi.*f).^n)+1./r));

Z_sig(2:end,:)=zbest-Z_cpe;
Z_res(2:end,:)=data(2:end,:)-Z_cpe;
end