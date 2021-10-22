function [para,fval,r2]=findfit2(f,zdata,x0)
lb=[0,0,-1e-3,2000,0.01,0,1e-8,1e-8];
ub=[1,1,1e-3,6000,4000,1,1e-3,1e-3];
[para,fval,r2] = two_component_fit2(f,zdata,x0);
r2_old=r2;
for i=1:1000
    para=para+(1-r2).*(rand(1,8).*(ub-lb)-(para-lb)).*0.0001;
    [parai,fvali,r2i] = two_component_fit2(f,zdata,para);
    fprintf('attempt%d,r2: %f',i,r2);
    if r2>r2_old
        para=parai;
        fval=fvali;
        r2=r2i;
    end 
end