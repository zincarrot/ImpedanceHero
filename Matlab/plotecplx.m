function plote=plotecplx(f,z,c0)
e=1./(1i*c0.*f.*z);
plote=plot(real(e),imag(e));
xlabel("\epsilon'");
ylabel("\epsilon''");
set(gca,'Ydir','reverse');
%%axis equal;
end