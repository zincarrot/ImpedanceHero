function [Z_res,Z_sig, pbest,zbest,fval]=fit_ICEC_novel_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    'p(C1,s(L1,R1,E2,I4))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [8e-10,4e-6,200,1e-5,0.8,0,0.0005,0.01,100], ...          % initial parameters
    [0,0,0,0,0.7,0.00,0.0005,0,100], ...             % lower boundary conditions for parameters
    [1e-9,1e-5,1000,1e-3,1,0,0.0005,10,100], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );

       %1-day parameters
     % [1e-10,1e-6,200,7e-6,0.78,0.05,0.01,1e-4], ...          % initial parameters
     % [0,0,0,0,0.7,0,0.01,0], ...             % lower boundary conditions for parameters
     % [1e-8,1e-4,1000,1e-3,1,1,0.01,1e-1], ...             % upper boundary conditions for parameters

f=data(1,:);
n=pbest(:,5);
Q=pbest(:,4);
C_s=pbest(:,1);
L=pbest(:,2);


Z_sig=data;
Z_car=1./((1./data(2:end,:))-2j*pi.*f.*C_s);
Z_car_=1./((1./zbest)-2j*pi.*f.*C_s);
Z_l=2j*pi.*f.*L;
Z_cpe=1./(Q.*(2j*pi.*f).^n);
Z_sig(2:end,:)=Z_car-Z_cpe-Z_l;
Z_res=Z_sig;
Z_res(2:end,:)=Z_car_-Z_cpe-Z_l;
end