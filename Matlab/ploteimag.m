function ploteimag(f,z)
e=1./(1i.*f.*z);
semilogx(f,imag(e));
ylabel("\epsilon''");
xlabel("f/Hz");
set(gca,'Ydir','reverse');
end