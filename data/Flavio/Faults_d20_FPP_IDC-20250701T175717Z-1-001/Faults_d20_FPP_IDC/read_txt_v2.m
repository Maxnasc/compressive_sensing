% % Function to import a PANDA-File into MATLAB
% % Technische Universitaet Dresden
% % Institute of Electrical Power Systems and High Voltage Engineering 
% % PANDA - equiPment hArmoNic DAtabase
% % Author: M. Sc. Ana Maria Blanco
% % ver. 1.0 (06/03/2012)
% % © Copyright reserved

%% DESCRIPTION 
% % txt2mat(filename) loads data from filename (Panda txt file) into the 
% % workspace. 

% % Example: 
% >> filename='C:\{name of subdirectoy}\ExamplePandaFile.txt';
% >> Daten = txt2mat(filename);

% % Daten is a structure with the information of the EUT, the harmonics and
% % the waveforms (if are available).


%% FUNCTION

function[Daten]=read_txt_v2(filename)

%% IMPORT FILE
DELIMITER = ' ';
HEADERLINES = 36;
Complete_file = importdata(filename, DELIMITER, HEADERLINES);

j=1;
for ii=1:length(Complete_file)
    if cell2mat(strfind(Complete_file(ii),'='))>=0
        div=regexp(Complete_file(ii), '=|;', 'split');
        Rows_with_data(j,1)=div{1,1}(1,1);
        Rows_with_data(j,2)=div{1,1}(1,2);
        j=j+1;
    end
end

for ii=1:length(Rows_with_data)
    if isempty(regexpi(Rows_with_data{ii,1},'Project'))==0        
        Daten.Project.Project=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'PI'))==0
        Daten.Project.PI=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Funding'))==0
        Daten.Project.Funding=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Category'))==0
        Daten.EUT.Category=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Sub_Category'))==0
        Daten.EUT.Sub_Category=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Unique_ID'))==0
        Daten.EUT.UniqueID=char(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Manufacturer'))==0
        Daten.EUT.Manufacturer=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Product_code'))==0
        Daten.EUT.Product_code=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Sell_Country'))==0
        Daten.EUT.Sell_Country=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Sell_Year'))==0
        Daten.EUT.Sell_Year=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Rated_Power'))==0
        Daten.EUT.Rated_Power=strtrim(Rows_with_data{ii,2});
        if strcmp(Daten.EUT.Rated_Power, 'NA')==0
            Daten.EUT.Rated_Power=str2double(Daten.EUT.Rated_Power);
        end
    elseif isempty(regexpi(Rows_with_data{ii,1},'Nominal_Frequency'))==0
        Daten.EUT.Nominal_Frequency=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Nominal_Voltage'))==0
        Daten.EUT.Nominal_Voltage=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Supply_Source'))==0
        Daten.EUT.Supply_Source=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Description'))==0
        Daten.EUT.Description=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Current_Harmonics'))==0
        harmonics_temp=char(Rows_with_data{ii,2});
        harmonics_temp=char(strread(harmonics_temp, '%s ','delimiter', '/'));
        Daten.Harmonics.Current=str2num(harmonics_temp);
        clear harmonics_temp
    elseif isempty(regexpi(Rows_with_data{ii,1},'Voltage_Harmonics'))==0
        harmonics_temp=char(Rows_with_data{ii,2});
        harmonics_temp=char(strread(harmonics_temp, '%s ','delimiter', '/'));
        Daten.Harmonics.Voltage=str2num(harmonics_temp);
        clear harmonics_temp
    elseif isempty(regexpi(Rows_with_data{ii,1},'Sampling_Rate'))==0
        Daten.Waveforms.Sampling_Rate=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data1_Attenuation'))==0
        Daten.Waveforms.Data1_Attenuation=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data1_Type'))==0
        Daten.Waveforms.Data1_Type=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data2_Attenuation'))==0
        Daten.Waveforms.Data2_Attenuation=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data2_Type'))==0
        Daten.Waveforms.Data2_Type=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data3_Attenuation'))==0
        Daten.Waveforms.Data3_Attenuation=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data3_Type'))==0
        Daten.Waveforms.Data3_Type=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data4_Attenuation'))==0
        Daten.Waveforms.Data4_Attenuation=str2double(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data4_Type'))==0
        Daten.Waveforms.Data4_Type=strtrim(Rows_with_data{ii,2});
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data1'))==0
        waveform_temp=char(Rows_with_data{ii,2});
        waveform_temp=char(strread(waveform_temp, '%s ','delimiter', ','));
        Daten.Waveforms.Signal1=str2num(waveform_temp);
        clear waveform_temp
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data2'))==0
        waveform_temp=char(Rows_with_data{ii,2});
        waveform_temp=char(strread(waveform_temp, '%s ','delimiter', ','));
        Daten.Waveforms.Signal2=str2num(waveform_temp);
        clear waveform_temp
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data3'))==0
        waveform_temp=char(Rows_with_data{ii,2});
        waveform_temp=char(strread(waveform_temp, '%s ','delimiter', ','));
        Daten.Waveforms.Signal3=str2num(waveform_temp);
        clear waveform_temp
    elseif isempty(regexpi(Rows_with_data{ii,1},'Data4'))==0
        waveform_temp=char(Rows_with_data{ii,2});
        waveform_temp=char(strread(waveform_temp, '%s ','delimiter', ','));
        Daten.Waveforms.Signal4=str2num(waveform_temp);
        clear waveform_temp
    end
    
end
end