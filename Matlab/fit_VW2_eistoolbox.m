function [Z_res,Z_sig, pbest,zbest,fval]=fit_VW2_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    'p(C1,s(R1,E2,C1,p(R1,E2)))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [1e-10,500,1e-5,0.7,5e-6,300,1e-5,0.7], ...          % initial parameters
    [0,0,0,0.5,0,0,0,0], ...             % lower boundary conditions for parameters
    [1e-5,2000,1e-4,0.75,2e-5,1000,1e-3,1], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );
f=data(1,:);
n=pbest(:,4);
Q=pbest(:,3);
C_s=pbest(:,1);
C=pbest(:,5);


Z_sig=data;
Z_car=1./((1./data(2:end,:))-2j*pi.*f.*C_s);
Z_car_=1./((1./zbest)-2j*pi.*f.*C_s);
Z_cpe=1./(Q.*(2j*pi.*f).^n);
Z_C=1./(C.*(2j*pi.*f));
Z_sig(2:end,:)=Z_car-Z_cpe-Z_C;
Z_res=Z_car_-Z_cpe-Z_C;
end