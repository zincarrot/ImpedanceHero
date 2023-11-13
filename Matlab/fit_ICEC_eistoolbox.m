function [Z_res,Z_sig, pbest,zbest,fval]=fit_ICEC_eistoolbox(data)

%% guess
% [r0,n0,Q0]=guessCPE(data(2,:),data(1,:));

%% fit
[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    'p(C1,s(L1,R1,E2,p(R1,C1,E2)))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [6e-10,6e-6,150,2e-5,0.85,20,1e-6,1e-5,0.5], ...          % initial parameters
    [0,0,0,0,0.5,0,0,0,0.5], ...             % lower boundary conditions for parameters
    [1e-9,1e-4,1000,1e-3,1,100,1e-3,1e-2,0.5], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );
f=data(1,:);
n=pbest(:,4);
Q=pbest(:,3);
C_s=pbest(:,1);


Z_sig=data;
Z_car=1./((1./data(2:end,:))-2j*pi.*f.*C_s);
Z_car_=1./((1./zbest)-2j*pi.*f.*C_s);
Z_cpe=1./(Q.*(2j*pi.*f).^n);
Z_sig(2:end,:)=Z_car-Z_cpe;
Z_res=Z_car_-Z_cpe;
end