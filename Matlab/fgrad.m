function m=fgrad(x,f,zdata)
    xcell=num2cell(x);
    [Q,a,ch,fc,kl,n,r,rs]=xcell{:};
    fcell=num2cell(f);
    [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15]=fcell{:};
    zcell=num2cell(zdata);
    [zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15]=zcell{:};
    m=grad(Q,a,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,kl,n,r,rs,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15);
end