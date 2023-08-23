clc, close all
clear
addpath("./functions/")
addpath("./data/")
%% Generate the tables

% import the data

bm = readmatrix("./data/time_pos_bm.csv");
fbm_0_1 = readmatrix("./data/time_pos_fbm_0_1.csv");
fbm_0_5 = readmatrix("./data/time_pos_fbm_0_5.csv");
fbm_0_9 = readmatrix("./data/time_pos_fbm_0_9.csv");
fbm_0_25 = readmatrix("./data/time_pos_fbm_0_25.csv");
fbm_0_75 = readmatrix("./data/time_pos_fbm_0_75.csv");


data=[bm.', fbm_0_1.', fbm_0_25.', fbm_0_5.', fbm_0_75.', fbm_0_9.'];
%% % generate the boxplot and store it

file_name = fullfile('.','figures','boxplot_angiogenesis.pdf');

boxplot(data,'Notch','on', ...
    'PlotStyle','compact','Labels',{'bm','H=0.1','H=0.25','H=0.5', ...
    'H=0.75','H=0.9'})

exportgraphics(gcf,file_name,'ContentType','vector')

%% Generate the media plot + sdv

file_name = fullfile('.','figures','bar_plot.pdf');

mean_std = [
[mean(bm),std(bm)];
[mean(fbm_0_1),std(fbm_0_1)];
[mean(fbm_0_25),std(fbm_0_25)];
[mean(fbm_0_5),std(fbm_0_5)];
[mean(fbm_0_75),std(fbm_0_75)];
[mean(fbm_0_9),std(fbm_0_9)]];

labels_bar = categorical({'bm','H=0.1','H=0.25','H=0.5', ...
    'H=0.75','H=0.9'});
cm = validatecolor(jet(6),'multiple');
b = bar(labels_bar,mean_std(:,1),'FaceColor','flat');


hold on
for i = 1:6
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
ylabel('Tiempo hasta 200 µm (h)')
hold off

removeToolbarExplorationButtons(gcf)
exportgraphics(gcf, file_name, 'ContentType','vector');



