function plotmreal(f,z)
m=1i.*f.*z;
semilogx(f,real(m));
ylabel("M'");
xlabel("f/Hz");
end