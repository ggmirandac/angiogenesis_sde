%% Generate the tables
clc, close all
clear
addpath("./functions/")
addpath("./data/")


% import the data stored in the .mat files
% current folder
d_data = fullfile('.',"data");
data_in_dir = {dir(d_data).name};
for data = data_in_dir
    splitting = split(data{1,1},'.');
    extension = char(splitting(2,1));
    if extension == "mat"
        load(fullfile(d_data, data))
    end 
 
end








%% % generate the boxplot and store it

box_f_name = fullfile('.','figures','boxplot_angiogenesis.pdf');
time_data = [time_0_bm, time_0_h_01, time_0_h_02, time_0_h_025, time_0_h_03, ...
    time_0_h_035, time_0_h_04, time_0_h_05, time_0_h_06, time_0_h_07, time_0_h_075, ...
    time_0_h_08, time_0_h_09].';

   
grp = [ones(size(time_0_bm.')); 2.*ones(size(time_0_h_01.'));...
    3.*ones(size(time_0_h_02.')); 4.*ones(size(time_0_h_025.'));...
    5.*ones(size(time_0_h_03.')); 6.*ones(size(time_0_h_035.'));...
    7.*ones(size(time_0_h_04.')); 8.*ones(size(time_0_h_05.'));...
    9.*ones(size(time_0_h_06.')); 10.*ones(size(time_0_h_07.'));...
    11.*ones(size(time_0_h_075.')); 12.*ones(size(time_0_h_08.'));...
    13.*ones(size(time_0_h_09.'))];
b = boxplot(time_data, grp,'Notch','on','Labels',{'B.M.', ...
    'H = 0.35','H = 0.4','H = 0.5', ...
    'H = 0.6','H = 0.7','H = 0.75','H = 0.8','H = 0.9'});
%{
,'Labels',{'B.M.', ...
    'H = 0.1','H = 0.2','H = 0.25','H = 0.3','H = 0.35','H = 0.4','H = 0.5', ...
    'H = 0.6','H = 0.7','H = 0.75','H = 0.8','H = 0.9'}
%}
ylabel('Time [h]')
xlabel('Type of correlation')
exportgraphics(gcf,box_f_name,'ContentType','vector')

%% Generate the media plot + sdv

file_name = fullfile('.','figures','bar_plot.pdf');

mean_std = [
[mean(time_0_bm),std(time_0_bm)];
[mean(time_0_h_01),std(time_0_h_01)];
[mean(time_0_h_02),std(time_0_h_02)];
[mean(time_0_h_025),std(time_0_h_025)];
[mean(time_0_h_03),std(time_0_h_03)];
[mean(time_0_h_035),std(time_0_h_035)];
[mean(time_0_h_04),std(time_0_h_04)];
[mean(time_0_h_05),std(time_0_h_05)];
[mean(time_0_h_06),std(time_0_h_06)];
[mean(time_0_h_07),std(time_0_h_07)];
[mean(time_0_h_075),std(time_0_h_075)];
[mean(time_0_h_08),std(time_0_h_08)]
[mean(time_0_h_09),std(time_0_h_075)]];

labels_bar = categorical({'bm','H = 0.1','H = 0.2','H = 0.25', 'H = 0.3', ...
    'H = 0.35','H = 0.4','H = 0.5','H = 0.6','H = 0.7', ...
    'H = 0.75','H = 0.8','H = 0.9'});
cm = validatecolor(jet(length(mean_std)),'multiple');

b = bar(labels_bar,mean_std(:,1),'FaceColor','flat');


hold on
for i = 1:length(labels_bar)
    b.CData(i,:)=cm(i,:);
end

% now we plot the error bars
stand_error = mean_std(:,2) / length(mean_std(:,2));
err_high = mean_std(:,1) + stand_error(:);
err_low = mean_std(:,1) - stand_error(:);
er = errorbar(labels_bar, mean_std(:,1), err_low, err_high);
er.Color = [0,0,0];
er.LineStyle = 'none';
er.LineWidth = 0.8;
ylabel('Time until 200 µm (h)')
hold off

removeToolbarExplorationButtons(gcf)
exportgraphics(gcf, file_name, 'ContentType','vector');

%%


values_t0_h_09 = unique(time_0_h_09, 'sorted');
indexes = [];
for i = int64(linspace(1, length( values_t0_h_09), 5))
    value = values_t0_h_09(i);
    index = find(Time_possition_h_09 == value, 1, "first");
    indexes = [indexes, index];
end 

% We plot the sprouts that behave in this ways in the 10 ways

figure(8)
cm = jet(length(indexes));
color = 1;
for jx = indexes

    X = Xdata_h_09{jx};
    name = ['Time =',num2str(Time_possition_h_09(jx))];
    x_movido = X(1,:) - (max(X(1,:)));
    plot3(x_movido,X(2,:), X(3,:),'Color', cm(color,:),'LineStyle','-','DisplayName',name, ...
        'LineWidth',2);
    color = color + 1;
    hold on
end
axis tight
legend show
xlabel('X position [µm]')
ylabel('Y position [µm]')
zlabel('Time [h]')
%zlabel('Time [h]')
title('Angiogenesis with H = 0.9')
% save sprouts
