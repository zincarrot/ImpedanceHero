function [para,fval,r2] = two_component_fit2(f,zdata,x0)
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0,0,-1e-3,2000,0.01,0,1e-12,0];
ub=[1,1,1e-3,6000,4000,1e9,1e-3,1e-1];
nonlcon=[];
options = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true,'StepTolerance',1e-16,'FunctionTolerance',1e-16,'MaxIterations',200,'MaxFunctionEvaluations',1000000);
fun=@(x)fmse2(x,f,zdata);
[para,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% [para,fval] = fminunc(fun,x0,options);
% [para,fval]=fminsearch(fun,x0);
r2=r2_ln(para);

clf;
plotecplx(f,zdata,1);
hold on;
plotecplx(f,zd(para),1);
shg;
hold off;

    function zd=zd(x)
        a=x(1);
        aep=x(2);
        ch=x(3);
        fc=x(4);
        fcep=x(5);
        kl=x(6);
        rs=x(7);
        rsep=x(8);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(a,aep,ch,f,fc,fcep,kl,rs,rsep)1./(kl + f.*pi.*(ch + rs./(((f.*1i)./fc).^(1 - a) + 1)).*2i) - ((((f.*1i)./fcep).^(1 - aep) + 1).*1i)/(2.*f.*rsep.*pi);
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zd=z(a,aep,ch,f,fc,fcep,kl,rs,rsep);
    end
    function r2=r2_ln(x)
        r2 = 1-sum(abs((zd(x)-zdata)/zdata).^2);
        disp(r2);
    end
end