function [para_best,fval_best,r2_best]=fit_RCRC_try(f,zdata,x0_ub)
    for i=1:100
        x0=rand(1,7).*x0_ub;
        fval_best=Inf;
        r2_best=0;
        [para,fval,r2]=fit_RCRC(f,zdata,x0);
        if fval_best>fval
            para_best=para;
            fval_best=fval;
            r2_best=r2;
        end
    end
    disp(r2_best);
end