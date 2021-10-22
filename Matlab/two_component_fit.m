function [para,fval,r2] = two_component_fit(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[1e-6,0,0,2000,1e-4,0.5,1,5e-7];
ub=[3e-5,1,1e-3,5000,1e-2,1,10000,1e-5];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true, 'StepTolerance',1e-16,'FunctionTolerance',1e-16,'MaxIterations',10000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse(x,f,zdata);
[para,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% [para,fval] = fminunc(@mse_ln,x0,options);
% [para,fval]=fminsearch(fun,x0);
r2=r2_ln(para);
    function r2=r2_ln(x)
        Q=x(1);
        a=x(2);
        ch=x(3);
        fc=x(4);
        kl=x(5);
        n=x(6);
        r=x(7);
        rs=x(8);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(Q,a,ch,f,fc,kl,n,r,rs)1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i);
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(Q,a,ch,f,fc,kl,n,r,rs);
        r2 = 1-sum(abs((zd-zdata)/zdata).^2);
        disp(r2);
    end
end