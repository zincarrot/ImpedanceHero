function [para,fval,r2] = fit_RCRC(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[1e-20,1e-10,1,1e4,-1,-1,0.1];
ub=[1e-7,1e-3,3000,1e9,-0.1,0,1];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true,'StepTolerance',1e-30,'FunctionTolerance',1e-30,'MaxIterations',3000,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse_RCRC(x,f,zdata);
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
        C1=x(1);
        C2=x(2);
        R1=x(3);
        R2=x(4);
        a=x(5);
        bb=x(6);
        c=x(7);
        zd=RCRC(R1,C1,R2,C2,a,bb,c,f);
    end
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end