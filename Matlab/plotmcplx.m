function plotmcplx(f,z)
m=1i*2*pi*f.*z;
plot(real(m),imag(m));
xlabel("M'");
ylabel("M''");
set(gca,'Ydir','reverse');
%%axis equal;
end