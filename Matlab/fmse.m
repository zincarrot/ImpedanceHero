function [m,fgrad]=fmse(x,f,zdata)
%     xcell=num2cell(x);
%     [Q,a,ch,fc,kl,n,r,rs]=xcell{:};
%     fcell=num2cell(f);
%     [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15]=fcell{:};
%     zcell=num2cell(zdata);
%     [zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15]=zcell{:};
    m=mse(x(1),x(2),x(3),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),f(11),f(12),f(13),f(14),f(15),x(4),x(5),x(6),x(7),x(8),zdata(1),zdata(2),zdata(3),zdata(4),zdata(5),zdata(6),zdata(7),zdata(8),zdata(9),zdata(10),zdata(11),zdata(12),zdata(13),zdata(14),zdata(15));
    if nargout > 1
        fgrad=grad(x(1),x(2),x(3),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),f(11),f(12),f(13),f(14),f(15),x(4),x(5),x(6),x(7),x(8),zdata(1),zdata(2),zdata(3),zdata(4),zdata(5),zdata(6),zdata(7),zdata(8),zdata(9),zdata(10),zdata(11),zdata(12),zdata(13),zdata(14),zdata(15));
    end
end
