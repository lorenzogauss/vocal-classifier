% --- Configurazione Parametri ---
fs = 44100;         
nfft = 2048;
tempoAnalisi = 10;
destHost = "127.0.0.1";
destPort = 5005;
blockSize = 64;
samplesToCollect = fs * tempoAnalisi;

% --- Gestione Porte UDP ---
oldPorts = udpportfind;
if ~isempty(oldPorts), delete(oldPorts); end
try
    u = udpport("LocalPort", 5006); 
catch
    nuovaPorta = 5007 + floor(rand*100);
    u = udpport("LocalPort", nuovaPorta);
end

% --- Inizializzazione Audio ---
samplesPerFrame = 4096;
deviceReader = audioDeviceReader('SampleRate', fs, 'SamplesPerFrame', samplesPerFrame); 
setup(deviceReader);

disp('>>> MATLAB: Sistema avviato. Invio continuo ogni 10 secondi.');

try
    while true
        audioBuffer = [];
        
        % 1. Accumulo Audio
        fprintf('Recording (10s)... ');
        while length(audioBuffer) < samplesToCollect
            audioIn = deviceReader();
            audioBuffer = [audioBuffer; audioIn];
        end
 
        % 2. Calcolo FFT
        fprintf('\nComputing FFT... ');
        Y = fft(audioBuffer, nfft);
        P2 = abs(Y/length(audioBuffer));
        P1 = P2(1:nfft/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        % 3. Normalizzazione e Invio (Sempre, anche se c'è silenzio)
        % Usiamo eps per evitare divisioni per zero se il buffer è totalmente muto
        features = double(P1 / max(P1 + eps))'; 
        
        for i = 1:blockSize:length(features)
            endIdx = min(i + blockSize - 1, length(features));
            chunk = features(i:endIdx);
            write(u, chunk, "double", destHost, destPort);
        end
        
        fprintf('Sent to Python!\n');
        
        drawnow limitrate;
    end
catch ME
    if exist('u', 'var'), delete(u); end
    release(deviceReader);
    fprintf('\n>>> Session ended. Error: %s\n', ME.message);
end