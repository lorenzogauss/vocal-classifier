fs = 44100;
winSamples = fs * 10;
nfft = 2048;

[audioM, fsM] = audioread('mina.mp3');
[audioC, fsC] = audioread('cammariere.mp3');

if fsM ~= fs, audioM = resample(audioM, fs, fsM); end
if fsC ~= fs, audioC = resample(audioC, fs, fsC); end

disp('Feature extraction - Mina');
datasetMina = extract_fft_features(audioM, winSamples, nfft, "Mina");

disp('Feature extraction - Cammariere');
datasetCamm = extract_fft_features(audioC, winSamples, nfft, "Cammariere");

fullDataset = [datasetMina; datasetCamm];
writetable(fullDataset, 'training_set.csv');
disp('Dataset ready');

% --- Feature extraction function ---
function T = extract_fft_features(signal, win, nfft, label)
    if size(signal,2) > 1, signal = mean(signal,2); end
    
    numSegments = floor(length(signal)/win);
    features = zeros(numSegments, nfft/2 + 1);
    
    for i = 1:numSegments
        idx = (i-1)*win + 1 : i*win;
        segment = signal(idx);
        Y = fft(segment, nfft);
        P2 = abs(Y/length(segment));
        P1 = P2(1:nfft/2+1);
        P1(2:end-1) = 2*P1(2:end-1); 
        P1 = P1 / max(P1 + eps); 
        features(i, :) = P1';
    end
    
    T = array2table(features);
    T.label = repmat(string(label), numSegments, 1);
end