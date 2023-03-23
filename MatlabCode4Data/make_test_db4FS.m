
% This module can generate test samples for full-resolution test from original PAN and LRMS images and save it with h5 format.
%
%% remark:
% @main_paths : directory of original PAN and LRMS images
% @sensor: sensor type to generate
% @channels: number of multispectral image channels
% @save_path: directory to save the generated samples
%% Output:
%

%   Author: Kyongson Jon (ksjhon911@gmail.com)
%   Date  : 11/18/2021

clc
clear variables
close all;

ratio = 4;%% resizing Factor

sensor = 'WV3'; channels = 8;
% sensor = 'QB'; channels = 4;
% sensor = 'WV2'; channels = 4;
% sensor = 'GF-2';channels = 4;

main_paths{1} = sprintf('%RawData/%s/', sensor); % {1} is reserved for multiple suites of database.

save_path = 'data';

dirInfo = dir(strcat(main_paths{1},'*.TIF'));
itemPairs = size(dirInfo, 1) / 2; % number of scenes, divided by 2 with consideration that each scene has 2 files for PAN and MS image.
for i = 1:itemPairs
    pan_files{1}{i} = strcat(string(i), '-PAN.TIF');
    mul_files{1}{i} = strcat(string(i), '-MUL.TIF');
end

if 0 %to avoid repeated data reading
    for data_case=1:1
        figure;
        for file_ind=1:size(pan_files{data_case},2)
            pan1 = imread(strcat(main_paths{data_case}, pan_files{data_case}{file_ind}));
            mul1 = imread(strcat(main_paths{data_case}, mul_files{data_case}{file_ind}));
            pans{data_case}{file_ind} = pan1; % all collection of PAN images
            muls{data_case}{file_ind} = mul1; % all collection of MS images
            
            %%%%show to confirm
            %figure;imshow(pan1, []);title(sprintf("%d-%d:%s", size(pan1, 1), size(pan1, 2), pan_files{data_case}{file_ind}));
            subplot(3, ceil(size(pan_files{data_case},2)/3), file_ind);
            imshow(imresize(pan1, [256 256]), []);title(sprintf("%d-%d:%s", size(pan1, 1), size(pan1, 2), pan_files{data_case}{file_ind}));
        end
    end
    
    save(sprintf('%s/all_raws(%s).mat', save_path, sensor), 'pans', 'muls');
else
    load(sprintf('%s/all_raws(%s).mat', save_path, sensor))
end
cnt_item = size(pans{1},2); % number of scenes

% elms: spatial sizes of all scenes
for i = 1:cnt_item
    elms(i) = numel(pans{1}{i});
end

clear PANS LRMSS USMSS
L = 64; % for result reported in tables
L = 128; % for drawing figures with clear visual effect
tot_cnt = 1000;
% kount_tot: number of samples from each scene, which is proportional to the spatial size of each scene
for i = 1:cnt_item - 1
    kount_tot(i) = uint32(elms(i)/sum(elms)*tot_cnt/20 )*20;
end
kount_tot(cnt_item) = tot_cnt - sum(kount_tot); % for the last scene, handly adjust

rand('state', sum(1000000*clock)); % randomly initialize

kount = kount_tot;
global PANS; %pan patches from downsampled PAN
global LRMSS; %ms patches from downsampled MS image
global USMSS; % upsampled LRMS using interp23tap
% firstly, prepare memory stacks to accelerate the sequential processing
% network input: PANS & LRMSS; target: HRMSS; USMSS: upsampled LRMS
PANS = zeros(sum(kount), 1, L*ratio, L*ratio);
USMSS = zeros(sum(kount), channels, L*ratio, L*ratio);
LRMSS = zeros(sum(kount), channels, L, L);

countEntered = 0; % number of generated samples
%%%%%%% make test data
for i = 1:cnt_item
    pan1 = pans{1}{i}; mul1 = muls{1}{i};
    create_tstData4FS(pan1, mul1, kount(i), L, ratio, sensor, countEntered);
    countEntered = countEntered + kount(i);
end
sample_index = 1:sum(kount);
tst_index = datasample(sample_index, 500,'Replace',false);

fname = sprintf('%s%stst_fs(%s)_%04d-%04d.h5', save_path, filesep, sensor, L, L);
fname = sprintf('%s%slarge_tst_fs(%s)_%04d-%04d.h5', save_path, filesep, sensor, L, L);
cur_index = tst_index;
% save samples to external storage
h5create(fname,'/PANS',size(PANS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20) ,1, L/2, L/2]);h5write(fname,'/PANS', single(PANS(cur_index,:,:,:)));
h5create(fname,'/LRMSS',size(LRMSS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20) ,channels / 2, L/2, L/2]);h5write(fname,'/LRMSS', single(LRMSS(cur_index,:,:,:)));
h5create(fname,'/USMSS',size(USMSS(cur_index,:,:,:)),'Deflate',9,'Datatype','single','ChunkSize',[double(size(cur_index,2)/20),channels / 2, L/2, L/2]);h5write(fname,'/USMSS', single(USMSS(cur_index,:,:,:)));

% end of processing
clear PANS HRMSS LRMSS USMSS

function create_tstData4FS(PAN, MS, kount, L, ratio, sensor, cur_count)
%%
%   To train ms-pan mapping, set input=PANS, output=HRMSS
%   To train pan-network, set input=PANS & LRMSS, output=HRMSS
%
%%
global PANS;
global LRMSS;
global USMSS;

tag = sensor;
if strcmp(sensor, 'GF-2') || strcmp(sensor, 'WV4')
    tag = 'none';
end
I_MS_LR = MS;
I_PAN = PAN;

n = size(I_MS_LR, 1);m = size(I_MS_LR, 2);

for i = 1: kount
    while 1 % to eliminate invalid patches(e.g., black patch)
        regLR_V = ceil(rand*(n-L+1))+(0:L-1); regLR_H = ceil(rand*(m-L+1))+(0:L-1);
        regHR_V = ratio * (regLR_V(1) - 1) + 1:ratio * (regLR_V(1) - 1) + ratio * L;
        regHR_H = ratio * (regLR_H(1) - 1) + 1:ratio * (regLR_H(1) - 1) + ratio * L;
        if  max(regHR_V) <= size(I_PAN, 1) && max(regHR_H) <= size(I_PAN, 2)
            crop_pan = I_PAN(regHR_V, regHR_H);
            h_hist = sum(crop_pan);v_hist = sum(crop_pan');
            if isempty( find(h_hist == 0) ) && isempty( find(v_hist == 0) )
                break;
            end
        end
    end
    
    crop_lrms = I_MS_LR(regLR_V, regLR_H,:);
    crop_usms = interp23tap(crop_lrms, ratio);
    cur_count = cur_count + 1;
    
    PANS(cur_count, 1, :, :) = crop_pan;
    LRMSS(cur_count, :, :, :) =  permute(crop_lrms, [3 1 2]);
    USMSS(cur_count, :, :, :) =  permute(crop_usms, [3 1 2]);
end
end