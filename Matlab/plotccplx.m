function plotccplx(f,z)
c=1./(1i*2*pi*f.*z);
plot(real(c),imag(c));
xlabel("C'/F");
ylabel("C''/F");
set(gca,'Ydir','reverse');
%%axis equal;
end