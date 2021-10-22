function [para,fval,r2] = zcc_fit(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0,-1e-5,1000,0];
ub=[1,1e-5,5000,1e-5];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true, 'StepTolerance',1e-20,'FunctionTolerance',1e-20,'MaxIterations',5000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse_zcc(x,f,zdata);
[para,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% [para,fval] = fminunc(@mse_ln,x0,options);
% [para,fval]=fminsearch(fun,x0);
clf;
plotecplx(f,zdata,1);
hold on;
plotecplx(f,zd(para),1);
shg;
hold off;
r2=r2_ln(para);
    function zd=zd(x)    
        a=x(1);
        ch=x(2);
        fc=x(3);
        rs=x(4);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(a, ch, fc, rs)-1i./(2.*f.*pi.*(ch + rs./(((f.*1i)./fc).^(1 - a) + 1)));
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(a, ch, fc, rs);
    end
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end