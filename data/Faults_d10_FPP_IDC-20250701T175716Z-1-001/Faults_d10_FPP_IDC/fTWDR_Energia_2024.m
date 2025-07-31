function [Energ, Energ_a, Energ_b] = fTWDR_Energia_2024(x, dk, wavelet, tipo)

%tipo = 1: coef. detalhes
%tipo = 0: coef. aproximacao

%Coeficientes dos filtros
[LO_R,HI_R,LO_D,HI_D] = wfilters(char(wavelet));
if tipo == 1
    filtro = HI_D/sqrt(2);
else
    filtro = LO_D/sqrt(2);
end
L = size(filtro, 2);

n1 = 1;
n2 = length(x);

%Apontador coeficiente
iw = 1;
%Apontador sinal
ix = n1+L-1;
%Apontador energia
ie = 1;

%Buffer com os coeficients ao quadrado em tempo real
Coef2 = zeros(dk, 1);

%Soma dos coeficientes ao quadrado
SCoef2 = 0;

%Calculo dos coeficients no primeiro ciclo - uma mostra
for i=ix:ix+dk-L-1
    Coef2(i-ix+L) = (filtro*x(i-L+1:i))^2;
    SCoef2 = SCoef2 + Coef2(i-ix+L);
end
iw = dk;
ix = n1+dk-1;

%Energia dos coeficientes
Energ = zeros(n2-n1+1,1);
Energ_a = zeros(n2-n1+1,1);
Energ_b = zeros(n2-n1+1,1);
for i=ix:n2
    SCoef2 = SCoef2 - Coef2(iw);
    Coef2(iw) = (filtro*x(i-L+1:i))^2;
    SCoef2 = SCoef2 + Coef2(iw);
    
    %Sinal modificado com efeito de borda
    Xm = [x(i-L+2:i); x(i-dk+1:i-dk+L)];
    
    %Coeficientes ao quadrado da TWDR com efeito de borda
    for j=L:2*L-2
        SCoef2 = SCoef2 - Coef2(j-L+1,1);
        Coef2(j-L+1,1) = (filtro*Xm(j-L+1:j))^2;
        SCoef2 = SCoef2 + Coef2(j-L+1,1);
        
        Energ_a(i) = Energ_a(i) + Coef2(j-L+1,1);
    end

    %Energia dos coeficientes
    Energ(i) = SCoef2;
    Energ_b(i) = SCoef2 - Energ_a(i);
   
    iw = iw +1;
    if iw>dk
        iw = L;
    end
end

