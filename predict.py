import socket
import joblib
import numpy as np
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
TARGET_SIZE = 1025 

try:
    model = joblib.load('classificatore_voci.pkl')
    classes = model.classes_
    print(f"Modello caricato con successo.")
    print(f"In attesa di dati FFT da 10 secondi (Target: {TARGET_SIZE} valori)...")
except Exception as e:
    print(f"ERRORE: Impossibile caricare il modello: {e}")
    exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

buffer = []

try:
    while True:
        data, addr = sock.recvfrom(4096)
        chunk = np.frombuffer(data, dtype=np.float64)
        

        buffer.extend(chunk.tolist())
        
        if len(buffer) >= TARGET_SIZE:
            current_features = np.array(buffer[:TARGET_SIZE]).reshape(1, -1)
            
            prediction = model.predict(current_features)[0]
            
            probs = model.predict_proba(current_features)[0]
            confidence = np.max(probs) * 100
            
            timestamp = time.strftime('%H:%M:%S')
            print(f"\n[{timestamp}] ANALISI COMPLETATA")
            print(f"   PREDIZIONE : {prediction.upper()}")
            print(f"   SICUREZZA : {confidence:.2f}%")
            print("-" * 30)
            buffer = [] 

except KeyboardInterrupt:
    print("Spegnimento del sistema in corso...")
finally:
    sock.close()