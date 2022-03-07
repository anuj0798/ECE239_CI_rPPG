%% Data Set 1
video_front = 'video_front_set1.mp4';
video_bottom = 'video_bottom_set1.mp4';
data1 = 'MPDataExport_set1.csv';
[RMSE, MAE] = POS_trial(video_front, video_bottom, data1);
%% Data Set 2
video_front2 = 'video_front.mp4';
video_bottom2 = 'video_bottom.mp4';
data2 = 'MPDataExport.csv';
[RMSE2, MAE2] = POS_trial(video_front2, video_bottom2, data2);
%% Graphs
X = categorical({'Set1 Data','Set2 Data'});
X = reordercats(X,{'Set1 Data', 'Set2 Data'});
Y = [RMSE, RMSE2];
figure(1)
bar(X,Y)
title('Performance of POS on Test Data')
xlabel('Data')
ylabel('RMSE')
Y1 = [MAE, MAE2];
figure(2)
bar(X,Y1)
title('Performance of POS on Test Data')
xlabel('Data')
ylabel('MAE')