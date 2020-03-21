clear
clc
close all
dat=csvread('../Result/rst_color.csv');
% [ndat,lambda]=boxcox(dat);
plot(dat,'+')