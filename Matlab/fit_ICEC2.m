function [para,fval,r2] = fit_ICEC2(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0,0,0];
ub=[1000,1,1];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true,'StepTolerance',1e-20,'FunctionTolerance',1e-16,'MaxIterations',2000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse_ICEC2(x,f,zdata);
[para,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% [para,fval] = fminunc(fun,x0,options);
% [para,fval]=fminsearch(fun,x0);
r2=r2_ln(para);

% clf;
% plotecplx(f,zdata,1);
% hold on;
% plotecplx(f,zd(para),1);
% shg;
% hold off;

clf;
plotcplx(zdata);
hold on;
plotcplx(zd(para));
shg;
hold off;

    function zd=zd(x)
        B=x(1);
        C=x(2);
        R=x(3);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(B,C,R)R + 1./(B.*(f*pi*2i).^C);
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(B,C,R);
    end
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end