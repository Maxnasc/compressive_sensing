% % Function to import a csv file from the oscilloscope ??? to PANDA File
% % Michigan Tech
% % Author: Flavio Costa, 25 December, 2023
% % ver. 3.0 (May 21, 2024)
% % © Copyright reserved
clc
clear all
close all

%current directory
directory = pwd();

%Original file name
file_name = 'tek0395';

%File name
file = sprintf('%s\\%s.txt',directory,'HVDC_2024_05_24_fs250M_d10_Fpg_WithCapacitors_01');

%Creating the new file
fid = fopen(file, 'w');

%Writing information on the file
fprintf(fid, '%s\n\n', 'Michigan Tech - Prof. Flavio B. Costa, 2024');

fprintf(fid, '%s\n', '[Project]');
fprintf(fid, '%s\n', 'Project = HVDC Project;');
fprintf(fid, '%s\n', 'PI = Flavio Bezerra Costa;');
fprintf(fid, '%s\n', 'Funding = 2023 Michigan Tech GLRC/ICC Rapid Seedling Grant;');

fprintf(fid, '%s\n', '[EUT]');
fprintf(fid, '%s\n', 'Category = 0;');
fprintf(fid, '%s\n', 'Sub_Category = 0;');
fprintf(fid, '%s\n', 'Manufacturer = Intertek;');
fprintf(fid, '%s%s%s\n', 'Product_code = HVDC Project, ',file_name,';');
fprintf(fid, '%s\n', 'Unique_ID = 0;');
fprintf(fid, '%s\n', 'Sell_Country = 0;');
fprintf(fid, '%s\n', 'Sell_Country = 0;');
fprintf(fid, '%s\n', 'Sell_Year = 0;');
fprintf(fid, '%s\n', 'Rated_Power = 0;');
fprintf(fid, '%s\n', 'Nominal_Frequency = 60;');
fprintf(fid, '%s\n', 'Nominal_Voltage = 0;');
fprintf(fid, '%s\n', 'Supply_Source = 0;');
fprintf(fid, '%s\n\n', 'HVDC testbed with diode-based rectifier, Inductors and capacitors between DC power stations and DC Lines, three-phase R load (R=100 Ohms) with LC filters, fault on 10% of the DC line, three-phase inverter with SPWM in the remote terminal, 4 measurements; DC voltage and current on local converter; DC voltage and current on remote converter.');

fprintf(fid, '%s\n', '[Waveforms]');
fprintf(fid, '%s\n', 'Sampling_Rate = 250000;');
fprintf(fid, '%s\n', 'Data1_Attenuation = 1;');
fprintf(fid, '%s\n', 'Data1_Type = Voltage;');
fprintf(fid, '%s\n', 'Data2_Attenuation = 1;');
fprintf(fid, '%s\n', 'Data2_Type = Voltage;');
fprintf(fid, '%s\n', 'Data3_Attenuation = 1;');
fprintf(fid, '%s\n', 'Data3_Type = Voltage;');
fprintf(fid, '%s\n', 'Data4_Attenuation = 1;');
fprintf(fid, '%s\n', 'Data4_Type = Voltage;');

%File from the oscilloscope
file_osc = sprintf('%s\\%s.csv',directory,file_name);
fid_osc = fopen(file_osc, 'r');

data1 = '';
data2 = '';
data3 = '';
data4 = '';

%Read the content
linha = fgetl(fid_osc);
t0=0;
t1=0;
fs=0;
bdata1 = false;
bdata2 = false;
bdata3 = false;
bdata4 = false;
while ischar(linha)
    % Process the content of each line as needed
    parts = strsplit(linha, ',');

    %only numbers. We need to get the data. Skipping label information
    %in the beginning
    if isempty(parts{1}) || isnan(str2double(parts{1}))
        linha = fgetl(fid_osc); % Read the next line
        continue; % Skip the rest of the loop iteration
    end

    if size(parts,2)>1
        trimmed_value = strtrim(parts{2}); % Remove leading and trailing spaces
        if ~isempty(parts{2})
            if isempty(data1)
                data1 = sprintf('%s', parts{2});
            else
                data1 = sprintf('%s,%s', data1, parts{2});
            end
            %computing the sampling rate
            if t0~=0 && t1==0
                t1 = str2double(parts{1});
                fs = 1/(t1-t0);
            end
            if t0==0 && t1==0
                t0 = str2double(parts{1});
            end
            if bdata1==false
                bdata1 = true;
            end
        end
    end
    if size(parts,2)>2
        if ~isempty(parts{3})
            if isempty(data2)
                data2 = sprintf('%s', parts{3});
            else
                data2 = sprintf('%s,%s', data2, parts{3});
            end
        end
        if bdata2==false
            bdata2 = true;
        end
    end
    if size(parts,2)>3
        if ~isempty(parts{4})
            if isempty(data3)
                data3 = sprintf('%s', parts{4});
            else
                data3 = sprintf('%s,%s', data3, parts{4});
            end
        end
        if bdata3==false
            bdata3 = true;
        end
    end
    if size(parts,2)>4
        if ~isempty(parts{5})
            if isempty(data4)
                data4 = sprintf('%s', parts{5});
            else
                data4 = sprintf('%s,%s', data4, parts{5});
            end
        end
        if bdata4==false
            bdata4 = true;
        end
    end
    
    linha = fgetl(fid_osc); % Read the next line
end

if bdata1==true
    fprintf(fid, '%s%s\n', 'Data1 = ',data1);
end
if bdata2==true
    fprintf(fid, '%s%s\n', 'Data2 = ',data2);
end
if bdata3==true
    fprintf(fid, '%s%s\n', 'Data3 = ',data3);
end
if bdata4==true
    fprintf(fid, '%s%s', 'Data4 = ',data4);
end

%Closing the files
fclose(fid);
fclose(fid_osc);



%Correcting the sampling rate

% Reopen the file in read mode
fid = fopen(file, 'r');
if fid == -1
    error('Failed to reopen the file for reading.');
end

% Read the file's contents into a cell array
fileContents = cell(0, 1);
while ~feof(fid)
    fileContents{end+1} = fgetl(fid);
end
fclose(fid);

% Find and update the 'Sampling_Rate' line
for i = 1:length(fileContents)
    if contains(fileContents{i}, 'Sampling_Rate')
        % Update the line with the calculated value fs
        updatedLine = sprintf('Sampling_Rate = %.2f;', fs);
        fileContents{i} = updatedLine;
        break; % Exit the loop once the line is updated
    end
end

% Reopen the file in write mode to rewrite the updated contents
fid = fopen(file, 'w');
if fid == -1
    error('Failed to reopen the file for writing.');
end

% Write the updated contents back to the file
fprintf(fid, '%s\n', fileContents{:});

% Close the file
fclose(fid);




