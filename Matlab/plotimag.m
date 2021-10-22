function plotimag(f,z)
semilogx(f,imag(z));
ylabel("Z''/\Omega");
xlabel("f/Hz");
set(gca,'Ydir','reverse');
end