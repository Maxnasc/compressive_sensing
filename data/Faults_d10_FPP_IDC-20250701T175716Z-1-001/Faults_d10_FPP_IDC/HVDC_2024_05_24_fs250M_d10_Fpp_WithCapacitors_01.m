%Michigan Tech
%Author: Flavio Costa, 25 December, 2023
%
clc
clear all
%close all

%Loading a file
[Daten]=read_txt_v2('HVDC_2024_05_24_fs250M_d10_Fpp_WithCapacitors_01.txt');

%Some parameters
f = Daten.EUT.Nominal_Frequency;%frequency
fs = Daten.Waveforms.Sampling_Rate;%Sampling frequency
nA = fs/f;%Samples per cycle
nA_real_int = ceil(nA);
nA_real_resto = nA - nA_real_int + 1;

% signal1 = Daten.Waveforms.Current*Daten.Waveforms.Data1_Attenuation;
% signal2 = Daten.Waveforms.Voltage*Daten.Waveforms.Data2_Attenuation;
% signal3 = Daten.Waveforms.Voltage*Daten.Waveforms.Data3_Attenuation;
v_remote = Daten.Waveforms.Signal1;
v_local = Daten.Waveforms.Signal2;
i_remote = -Daten.Waveforms.Signal3;
i_local = Daten.Waveforms.Signal4;
N = size(v_remote,1);



c = 3e+8;
%velocity = 1.81e+08;
velocity = 1.7e+08;
velocity/c

%Line length in meters
dLine = 3000;

%Transit time
tau = dLine/velocity;
ktau = floor(tau*fs);

%Fault distance: 10% from rectifier
df = 0.1*dLine;

kFL1=18424;
k0 = kFL1 - (0.1*dLine)*fs/velocity;%local
kFL2 = k0 + (0.3*dLine)*fs/velocity;%local
kFL3 = k0 + (0.5*dLine)*fs/velocity;%local

kFR1 = k0 + (0.9*dLine)*fs/velocity;%remote
kFR2 = k0 + (1.1*dLine)*fs/velocity;%remote
kFR3 = k0 + (1.3*dLine)*fs/velocity;%remote


figure(1)
plot(v_remote(1:N),'LineWidth',1,'LineStyle','-','Color',[0 0 1])
hold on
plot(v_local(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
%plot([1:N], signal1(1:N),'or')
hold off
xlim([1 N])
%xlim([kF-2*nA kF+8*nA])
ax = gca;
%ax.XTick = [0:ktau/10:N];
ax.XTickLabel = {'', '', '0', '1', '2','3','4','5','6','7','8','9','10'};
ylabel('Voltage')
%ylim([-50 50])
%ax.YTick = [-9.0:7.0];
%legend ('V DC Rectifier', 'I DC Inverter')
grid on
%box off


figure(2)
plot(1*i_remote(1:N),'LineWidth',1,'LineStyle','-','Color',[0 1 0])
hold on
plot(i_local(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
plot([k0 k0], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'k')
plot([kFL1 kFL1], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'r')
plot([kFL2 kFL2], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'r')
plot([kFL3 kFL3], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'r')
plot([kFR1 kFR1], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'g')
plot([kFR2 kFR2], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'g')
plot([kFR3 kFR3], [min(min(i_remote(1:N)),min(i_local(1:N))) max(max(i_remote(1:N)),max(i_local(1:N)))], 'g')
%plot([1:N], signal1(1:N),'or')
hold off
xlim([1 N])
%xlim([kF-2*nA kF+8*nA])
ax = gca;
%ax.XTick = [0:ktau/10:N];
ax.XTickLabel = {'', '', '0', '1', '2','3','4','5','6','7','8','9','10'};
ylabel('Current')
%ylim([-50 50])
%ax.YTick = [-9.0:7.0];
%legend ('V DC Rectifier', 'I DC Inverter')
grid on
%box off


%dk = 1000;%(40e-6 s)/10
dk = floor(2*ktau/10);%10dk = 2ktau - 2 transit time
figure(4)
plot([k0 k0], [-10 100], 'k')
hold on
plot([kFL1 kFL1], [-10 100], 'r')%local
plot([kFL2 kFL2], [-10 100], 'r')%local
plot([kFL3 kFL3], [-10 100], 'r')%local
plot(i_local(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
%plot([1:N], signal1(1:N),'or')
hold off
xlim([k0-dk k0+9*dk])
time = 10*dk/fs
ax = gca;
ax.XTick = [k0-dk:dk/2:k0+9*dk];
ax.XTickLabel = {'0', '1', '2','3','4','5','6','7','8','9','10'};
ylabel('Current')
ylim([-1 5])
%ax.YTick = [-9.0:7.0];
%legend ('V DC Rectifier', 'I DC Inverter')
grid on
%box off

figure(5)
plot([k0 k0], [-10 100], 'k')
hold on
plot([kFR1 kFR1], [-10 100], 'b')%remote
plot([kFR2 kFR2], [-10 100], 'b')%remote
plot([kFR3 kFR3], [-10 100], 'b')%remote
plot(i_remote(1:N),'LineWidth',1,'LineStyle','-','Color',[0 0 1])
%plot([1:N], signal1(1:N),'or')
hold off
xlim([k0-dk k0+9*dk])
time = 10*dk/fs
ax = gca;
ax.XTick = [k0-dk:dk/2:k0+9*dk];
ax.XTickLabel = {'0', '1', '2','3','4','5','6','7','8','9','10'};
ylabel('Current')
ylim([-1 5])
%ax.YTick = [-9.0:7.0];
%legend ('V DC Rectifier', 'I DC Inverter')
grid on
%box off


