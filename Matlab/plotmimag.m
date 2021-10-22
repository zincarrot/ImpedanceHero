function plotmimag(f,z)
m=1i.*f.*z;
semilogx(f,imag(m));
ylabel("M''");
xlabel("f/Hz");
set(gca,'Ydir','reverse');
end