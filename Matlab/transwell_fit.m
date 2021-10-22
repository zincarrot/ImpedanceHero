function [para,fval,r2] = transwell_fit(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0,0,0,0];
ub=[1,1,100000,11e8];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true, 'StepTolerance',1e-25,'FunctionTolerance',1e-25,'MaxIterations',10000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse_trans(x,f,zdata);
[para,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% [para,fval] = fminunc(@mse_ln,x0,options);
% [para,fval]=fminsearch(fun,x0);
clf;
plotecplx(f,zdata,1);
hold on;
plotecplx(f,zd(para),1);
shg;
hold off;
function zd=zd(x)
        ce=x(1);
        cw=x(2);
        rl=x(3);
        rw=x(4);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(ce,cw,rl,rw) rl + 1./(1./rw + pi.*cw.*f.*2i) - 1i./(2.*ce.*f.*pi);
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(ce,cw,rl,rw);
    end
r2=r2_ln(para);
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end