function [Z_sig,Z_res,pbest,zbest,fval]=fit_VW_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    's(R1,E2,p(C1,s(R1,C1)))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [200,5e-5,0.6,1.5e-6,1,4e-7], ...          % initial parameters
    [10,0,0.5,0,0,0], ...             % lower boundary conditions for parameters
    [500,1e-3,0.7,1e-4,1000,1e-2], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );
f=data(1,:);
n=pbest(:,4);
Q=pbest(:,3);
% C_s=pbest(:,1);


Z_sig=data;
Z_res=data;


% Z_car=1./((1./data(2:end,:))-2j*pi.*f.*C_s);
% Z_car_=1./((1./zbest)-2j*pi.*f.*C_s);
Z_cpe=1./((Q.*(2j*pi.*f).^n));

Z_sig(2:end,:)=zbest-Z_cpe;
Z_res(2:end,:)=data(2:end,:)-Z_cpe;
end