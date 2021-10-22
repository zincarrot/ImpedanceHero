function plotyimag(f,z)
y=1./z;
semilogx(f,imag(y));
ylabel("Y''/\Omega^{-1}");
xlabel("f/Hz");
set(gca,'Ydir','reverse');
end