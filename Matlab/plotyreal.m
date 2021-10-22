function plotyreal(f,z)
y=1./z;
semilogx(f,real(y));
ylabel("Y'/\Omega^{-1}");
xlabel("f/Hz");
end