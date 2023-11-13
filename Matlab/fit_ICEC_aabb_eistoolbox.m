function [Z_res,Z_sig, pbest,zbest,fval]=fit_ICEC_aabb_eistoolbox(data)

[pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    'p(C1,s(R1,P4,p(R1,C1)))', ...  % circuit model (string representation)
    0, ...            % lower frequency bound
    inf, ...            % upper frequency bound
    [1e-11,150,4000,0.8,6000,0.8,30,1e-6], ...          % initial parameters
    [0,0,0,0,0,0,0,0], ...             % lower boundary conditions for parameters
    [1e-5,1000,10000,1,10000,1,1000,1e-4], ...             % upper boundary conditions for parameters
    1, ...      % algorithm number
    1, ...      % weighting type number
    2000 ...         % max number of iterations
    );
f=data(1,:);
a=pbest(:,4);
A=pbest(:,3);
b=pbest(:,6);
B=pbest(:,5);
C_s=pbest(:,1);


Z_sig=data;
Z_car=1./((1./data(2:end,:))-2j*pi.*f.*C_s);
Z_car_=1./((1./zbest)-2j*pi.*f.*C_s);
Z_cpe=A./(f.^a)-1i*B./(f.^b);
Z_sig(2:end,:)=Z_car-Z_cpe;
Z_res=Z_car_-Z_cpe;
end