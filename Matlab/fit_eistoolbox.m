function [pbest,zbest,fval]=fit_eistoolbox( ...
    data, ...           % input data
    circuitstring, ...  % circuit model (string representation)
    lfb, ...            % lower frequency bound
    ufb, ...            % upper frequency bound
    pbest0, ...          % initial parameters
    LB, ...             % lower boundary conditions for parameters
    UB, ...             % upper boundary conditions for parameters
    algorithm, ...      % algorithm number
    weighting, ...      % weighting type number
    maxiter ...         % max number of iterations
    )

fdata=data(1,:);
data_idx=(fdata>=lfb)&(fdata<=ufb);
fdata_select=fdata(data_idx);
ndata=size(data,1)-1;

pbest=NaN(ndata,size(pbest0,2));
zbest=NaN(ndata,size(fdata_select,2));
fval=NaN(ndata);

% fprintf("computing:%d\n",1)
%     data1(:,1)=fdata_select';
%     data1(:,2)=real(data(3,data_idx))';
%     data1(:,3)=imag(data(3,data_idx))';
%     [pbest1,zbest1,fval1]= fitting_engine( ...
%         data1, ...           % input data
%         circuitstring, ...  % circuit model (string representation)
%         pbest0, ...          % initial parameters
%         LB, ...             % lower boundary conditions for parameters
%         UB, ...             % upper boundary conditions for parameters
%         1, ...      % algorithm number
%         weighting, ...      % weighting type number
%         10000 ...         % max number of iterations
%         );
%     pbest(1,:)=pbest1;
%     zbest(1,:)=zbest1(:,1)+1j*zbest1(:,2);
%     fval(1)=fval1;

for i=1:ndata
    if any(isnan(data(i+1,data_idx)))
        continue
    end
    fprintf("computing:%d",i)
    data1(:,1)=fdata_select';
    data1(:,2)=real(data(i+1,data_idx))';
    data1(:,3)=imag(data(i+1,data_idx))';
    [pbest1,zbest1,fval1]= fitting_engine( ...
        data1, ...           % input data
        circuitstring, ...  % circuit model (string representation)
        pbest0, ...          % initial parameters
        LB, ...             % lower boundary conditions for parameters
        UB, ...             % upper boundary conditions for parameters
        algorithm, ...      % algorithm number
        weighting, ...      % weighting type number
        maxiter ...         % max number of iterations
        );
    if fval1>5e-4
        continue
    end
    pbest0=pbest1;
    pbest(i,:)=pbest1;
    zbest(i,:)=zbest1(:,1)+1j*zbest1(:,2);
    fval(i)=fval1;
    fprintf("   fval=%d\n",fval1)
end
end