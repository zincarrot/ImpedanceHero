function plotycplx(z)
y=1./z;
plot(real(y),imag(y));
xlabel("Y'/\Omega^{-1}");
ylabel("Y''/\Omega^{-1}");
set(gca,'Ydir','reverse');
axis equal;
end