function kl=kl(x,f)
        Q=x(1);
        a=x(2);
        ch=x(3);
        fc=x(4);
        kl=x(5);
        n=x(6);
        r=x(7);
        rs=x(8);
%         lnz = @(Q,a,ch,f,fc,kl,n,r,rs)log(1.0./(Q.*(f.*pi.*2.0i).^n+1.0./r)+1.0./(kl+f.*pi.*(ch+rs./(((f.*1i)./fc).^(-a+1.0)+1.0)).*2.0i));
        z=@(Q,a,ch,f,fc,kl,n,r,rs)kl;
%         lnzd=lnz(Q,a,ch,f,fc,kl,n,r,rs);
        zcpe=z(Q,a,ch,f,fc,kl,n,r,rs);
end