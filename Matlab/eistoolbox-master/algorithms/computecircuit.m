%% COMPUTECIRCUIT
% This function takes the 'circuitstring' and parses it to obtain the 
% impedance of the circuit as a function of the frequency. z = f(freq).

% This is done by identifying each element (R1, C1, etc) and computing the
% impedance of each element separately (using the circuit element functions
% below).

% The impedance behavior of each element is stored in a column of z.

% Finally the total impedance is calculated, table z is destroyed and the
% impedance response of the circuit is presented at the output of the
% function.

% License information: 
% This file is modified from the original 'Zfit.m' library, to include only
% the sections and options used in 'eistoolbox.m'. Date: 15.11.2016
% The original file is Copyright (c) 2005 Jean-Luc Dellis; can be found in:
% http://de.mathworks.com/matlabcentral/fileexchange/19460-zfit

function z=computecircuit(param,circuit,freq)
    
    % Computes the complex impedance Z of the circuit string
    % process CIRCUIT to get the elements and their numeral inside CIRCUIT
    A=circuit~='p' & circuit~='s' & circuit~='(' & circuit~=')' & circuit~=',';
    element=circuit(A);

    k=0;
    % for each element
    for i=1:2:length(element-2)
        k=k+1;
        nlp=str2double(element(i+1));% idendify its numeral
        localparam=param(1:nlp);% get its parameter values
        param=param(nlp+1:end);% remove them from param
        
        % compute the impedance of the current element for all the frequencies
        z(:,k)=eval([element(i),'([',num2str(localparam),']',',freq)']);
        
        % modify the initial circuit string (to use it later with eval)
        circuit=regexprep(circuit,element(i:i+1),['z(:,',num2str(k),')'],'once');
    end
    
    z=eval(circuit);        % compute the global impedance
    z=[real(z),imag(z)];    % real and imaginary parts are separated to be processed

end % END of COMPUTECIRCUIT

%% CIRCUIT ELEMENT FUNCTIONS
% Calculate the impedance response of a single element
function z=R(p,f);  z=p*ones(size(f));   end    % Resistor
function z=C(p,f);  z=1./(1i*2*pi*f*p);  end    % Capacitor
function z=L(p,f);  z=1i*2*pi*f*p;       end    % Inductor
function z=E(p,f);  z=1./(p(1)*(1i*2*pi*f).^p(2)); end % CPE
function z=D(p,f); z=1./(2i*pi*f.*(p(1)+p(2)./(1+(2i*pi*f.*p(3)).^(1-p(4))))); end 
    %Cole-Cole, 1:eh 2:relaxation strength 3:tau 4:alpha
function z=P(p,f); z=p(1)./(f.^p(2))-1i.*p(3)./(f.^p(4));   end % more general EP model than CPE. 1:A,2:a,3:B,4:b
function z=I(p,f); z=1./((p(1)*sqrt((2i*pi*f*p(3)+p(4))))+(p(2)*(2i*pi*f*p(3)+p(4)))); end % ICEC model. 1: dp/rw, 2: lambda/dp, 3: tau
% Add multiple elements in series
function z=s(varargin)
    z = 0;
    for idx = 1:nargin
       z = z + varargin{idx};
    end
end  

% Add multiple elements in parallel
function z=p(varargin)
    z = 0;
    for idx = 1:nargin
       z = z + (1 ./ varargin{idx}) ;
    end
    z = 1 ./ z;
end       
