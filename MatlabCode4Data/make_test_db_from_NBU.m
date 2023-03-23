% This module generates test samples for P3Net from NBU dataset and save it with h5 format.
%
%% remark:
% @test_data_dir : directory of NBU dataset
% @itemPairs: given number of pairs
% @sensor: sensor type to generate
% @channels: number of multispectral image channels
% @save_path: directory to save the generated samples
%% Output:
%

%   Author: Kyongson Jon (ksjhon911@gmail.com)
%   Date  : 11/18/2021

clc
clear variables
close all

sensor = 'WV4'; channels = 4;
%sensor = 'WV3'; channels = 8;
%sensor = 'WV2'; channels = 8;
test_data_dir = '/NBU-Dataset/Satellite_Dataset/Dataset/4 WorldView-4/';
%test_data_dir = 'NBU-Dataset/Satellite_Dataset/Dataset/6 WorldView-3/';
%test_data_dir = 'NBU-Dataset/Satellite_Dataset/Dataset/5 WorldView-2/';
save_path = 'data';
itemPairs = 500; % given number of pairs for WV2
% temPairs = 160;  % given number of pairs for WV3
ratio = 4; %% resizing Factor
for i = 1:itemPairs
    pan_files{i} = strcat('PAN_1024/', string(i), '.mat');
    mul_files{i} = strcat('MS_256/', string(i), '.mat');
end

if 0   % to avoid repeated data reading
    for file_ind=1:itemPairs
        load(strcat(test_data_dir, pan_files{file_ind}));
        
        load(strcat(test_data_dir, mul_files{file_ind}));
        pans{file_ind} = imgPAN; % all collection of PAN images
        muls{file_ind} = imgMS; % all collection of MS images
        
        %%%% show to confirm
        %figure;imshow(imgPAN, []);title(sprintf("%d-%d:%s", size(imgPAN, 1), size(imgPAN, 2), pan_files{file_ind}));
    end
    
    save(sprintf('%s/all_NBU_raws(%s).mat', save_path, sensor), 'pans', 'muls');
else
    load(sprintf('%s/all_NBU_raws(%s).mat', save_path, sensor))
end

kount = itemPairs; % number of pairs to generate
L = 64; % for generalization test, this is maximum patch size.
global PANS;  %pan patches from downsampled PAN
global HRMSS; %pseudo-hrms patches, obtained from original lrms image, used to train MS-PAN mapping
global LRMSS; %ms patches from downsampled MS image
global USMSS; %upsampled LRMS using interp23tap

% firstly, prepare memory stacks to accelerate the sequential processing
% network input: PANS & LRMSS; target: HRMSS; USMSS: upsampled LRMS

PANS = zeros(sum(kount), 1, L*ratio, L*ratio);
HRMSS = zeros(sum(kount), channels, L*ratio, L*ratio);
USMSS = zeros(sum(kount), channels, L*ratio, L*ratio);
LRMSS = zeros(sum(kount), channels, L, L);

countEntered = 0;   % number of generated samples
%%%%%%% make training/validation data
for i = 1:kount
    pan1 = pans{i}; mul1 = muls{i};
    create_Data(pan1, mul1, 1, L, ratio, sensor, countEntered);
    countEntered = countEntered + 1;
end

fname = sprintf('%s%sNBU_tst_large(%s)_%04d-%04d.h5', save_path, filesep, sensor, L, L);
cur_index = 1:kount;

% save samples to external storage
h5create(fname,'/PANS',size(PANS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20) ,1, L/2, L/2]);h5write(fname,'/PANS', single(PANS(cur_index,:,:,:)));
h5create(fname,'/LRMSS',size(LRMSS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20) ,channels / 2, L/2, L/2]);h5write(fname,'/LRMSS', single(LRMSS(cur_index,:,:,:)));
h5create(fname,'/HRMSS',size(HRMSS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20) ,channels / 2, L/2, L/2]);h5write(fname,'/HRMSS', single(HRMSS(cur_index,:,:,:)));
h5create(fname,'/USMSS',size(USMSS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20),channels / 2, L/2, L/2]);h5write(fname,'/USMSS', single(USMSS(cur_index,:,:,:)));

% end of processing
clear PANS HRMSS LRMSS USMSS

%
function create_Data(PAN, MS, kount, L, ratio, sensor, cur_count)
%%

global PANS;
global LRMSS;
global HRMSS;
global USMSS;

tag = sensor;
if strcmp(sensor, 'GF-2') || strcmp(sensor, 'WV4')
    tag = 'none';
end
[I_MS_LR, I_PAN]=resize_images(MS,PAN,ratio,tag);

n = size(I_MS_LR, 1);m = size(I_MS_LR, 2);

for i = 1: kount
    regLR_V = ceil(rand*(n-L+1))+(0:L-1); regLR_H = ceil(rand*(m-L+1))+(0:L-1);
    regHR_V = ratio * (regLR_V(1) - 1) + 1:ratio * (regLR_V(1) - 1) + ratio * L;
    regHR_H = ratio * (regLR_H(1) - 1) + 1:ratio * (regLR_H(1) - 1) + ratio * L;
    crop_pan = I_PAN(regHR_V, regHR_H);
    
    crop_lrms = I_MS_LR(regLR_V, regLR_H,:);
    crop_hrms = MS(regHR_V, regHR_H,:);
    crop_usms = interp23tap(crop_lrms, ratio);
    cur_count = cur_count + 1;
    
    PANS(cur_count, 1, :, :) = crop_pan;
    HRMSS(cur_count, :, :, :) = permute(crop_hrms, [3 1 2]);
    LRMSS(cur_count, :, :, :) =  permute(crop_lrms, [3 1 2]);
    USMSS(cur_count, :, :, :) =  permute(crop_usms, [3 1 2]);
end
end
