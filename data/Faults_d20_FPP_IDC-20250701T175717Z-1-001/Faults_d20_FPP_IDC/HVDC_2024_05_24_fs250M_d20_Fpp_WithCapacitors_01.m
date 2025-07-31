%Michigan Tech
%Author: Flavio Costa, 25 December, 2023
%
clc
clear all
%close all

%Loading a file
[Daten]=read_txt_v2('HVDC_2024_05_24_fs250M_d20_Fpp_WithCapacitors_01.txt');

%Some parameters
f = Daten.EUT.Nominal_Frequency;%frequency
fs = Daten.Waveforms.Sampling_Rate;%Sampling frequency
nA = fs/f;%Samples per cycle
nA_real_int = ceil(nA);
nA_real_resto = nA - nA_real_int + 1;

% signal1 = Daten.Waveforms.Current*Daten.Waveforms.Data1_Attenuation;
% signal2 = Daten.Waveforms.Voltage*Daten.Waveforms.Data2_Attenuation;
% signal3 = Daten.Waveforms.Voltage*Daten.Waveforms.Data3_Attenuation;
signal1 = Daten.Waveforms.Signal1;
signal2 = Daten.Waveforms.Signal2;
signal3 = -Daten.Waveforms.Signal3;
signal4 = Daten.Waveforms.Signal4;
N = size(signal1,1);

T = 1/2*[1 1; 1 -1];
signal4 = signal4

c = 3e+8;
%velocity = 1.81e+08;
velocity = 1.65e+08;
velocity/c

%Line length in meters
dLine = 3000;

%Transit time
tau = dLine/velocity;
ktau = floor(tau*fs);

%Fault distance: 20% from rectifier
df = 0.2*dLine;

kFL1=36833;
k0 = kFL1 - (0.2*dLine)*fs/velocity;%local
kFL2 = k0 + (0.6*dLine)*fs/velocity;%local
kFL3 = k0 + (1.0*dLine)*fs/velocity;%local

kFR1 = k0 + (0.8*dLine)*fs/velocity;%remote
kFR2 = k0 + (1.2*dLine)*fs/velocity;%remote
kFR3 = k0 + (1.6*dLine)*fs/velocity;%remote


figure(1)
plot(signal1(1:N),'LineWidth',1,'LineStyle','-','Color',[0 0 1])
hold on
plot(signal2(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
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
hold on
plot(1*signal3(1:N),'LineWidth',1,'LineStyle','-','Color',[0 1 0])
plot(signal4(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
plot([k0 k0], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'k')
plot([kFL1 kFL1], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'r')
plot([kFL2 kFL2], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'r')
plot([kFL3 kFL3], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'r')
plot([kFR1 kFR1], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'g')
plot([kFR2 kFR2], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'g')
plot([kFR3 kFR3], [min(min(signal3(1:N)),min(signal4(1:N))) max(max(signal3(1:N)),max(signal4(1:N)))], 'g')
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

figure(3)
plot(1*signal3(1:N),'LineWidth',1,'LineStyle','-','Color',[0 1 0])
hold on
plot(signal4(1:N),'LineWidth',1,'LineStyle','-','Color',[1 0 0])
hold off
xlim([kFL1-250 kFL1+750+1500])
time = 500/fs
ax = gca;
ax.XTick = [kFL1-250:100:kFL1+750+1500];
ax.XTickLabel = {'0', '1', '2','3','4','5','6','7','8','9','10'};
ylabel('Voltage')
ylim([0 2])
%ax.YTick = [-9.0:7.0];
%legend ('V DC Rectifier', 'I DC Inverter')
grid on
%box off

