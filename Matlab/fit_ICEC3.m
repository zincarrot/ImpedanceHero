function [para,fval,r2] = fit_ICEC3(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[1000,0,0,0,20,0];
ub=[100000,1,1000,0.01,500,1];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true,'StepTolerance',1e-16,'FunctionTolerance',1e-16,'MaxIterations',1000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse_ICEC3(x,f,zdata);
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
        AA=x(1);
        B=x(2);
        C=x(3);
        R=x(5);
        Qcpe=x(4);
        ncpe=x(6);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(AA,B,C,R,Qcpe,ncpe)2*R + 1./(Qcpe*(f*pi*2i).^ncpe) - (AA*(C + pi*B*f*2i).^(1/2)*1i)./(2*B*f*pi);
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(AA,B,C,R,Qcpe,ncpe);
    end
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end