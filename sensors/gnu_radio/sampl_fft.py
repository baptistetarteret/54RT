import numpy as np
import matplotlib.pyplot as plt
import csv

def generate_radar_signal(
    f1, f2, 
    DI_list, DTOA_list, 
    modulation_type, 
    sampling_rate=1e5, 
    output_csv="radar_signal.csv"
):
    """
    Génère un signal radar selon le type de modulation spécifié.
    Modulations disponibles :
    - 'impulsionnel' : Une seule impulsion gaussienne pendant DI.
    - 'FMCW' : Variation de fréquence continue, rampe triangulaire (f1->f2->f1) sur tout DI.
    - 'CW' : Une seule fréquence constante pendant DI.
    - 'chirp' : Un seul chirp linéaire f1->f2 pendant DI.

    Retourne (t, signal_emit, signal_received, T_emit, dtoa).
    """

    # Sélection aléatoire de DI et DTOA
    idx = np.random.randint(0, len(DI_list))
    T_emit = DI_list[idx]
    dtoa = DTOA_list[idx]

    T_total = dtoa
    t = np.arange(0, T_total, 1 / sampling_rate)

    signal_emit = np.zeros_like(t)
    signal_duration = int(T_emit * sampling_rate)

    if modulation_type == 'impulsionnel':
        # Une seule impulsion gaussienne pendant DI
        f_random = np.random.uniform(f1, f2)
        t_imp = t[:signal_duration]
        seg_duration = T_emit
        t_centered = t_imp - (seg_duration / 2)
        sigma = T_emit / 6.0
        envelope = np.exp(-t_centered**2 / (2 * sigma**2))
        impulse_signal = envelope * np.sin(2 * np.pi * f_random * t_imp)
        signal_emit[:signal_duration] = impulse_signal

    elif modulation_type == 'FMCW':
        # FMCW continu : rampe triangulaire sur tout le DI (f1->f2->f1)
        t_seg = t[:signal_duration]
        seg_duration = T_emit
        half = seg_duration / 2
        f_inst = np.zeros(signal_duration)
        for i in range(signal_duration):
            tau = i / sampling_rate
            if tau < half:
                f_inst[i] = f1 + (f2 - f1)*(tau/half)
            else:
                f_inst[i] = f2 - (f2 - f1)*((tau - half)/half)
        phase = np.cumsum(2 * np.pi * f_inst / sampling_rate)
        fmcw_signal = np.sin(phase)
        signal_emit[:signal_duration] = fmcw_signal

    elif modulation_type == 'CW':
        # CW : une seule fréquence constante pendant DI
        f_random = np.random.uniform(f1, f2)
        t_seg = t[:signal_duration]
        cw_signal = np.sin(2 * np.pi * f_random * t_seg)
        signal_emit[:signal_duration] = cw_signal

    elif modulation_type == 'chirp':
        # Chirp : un seul chirp linéaire f1->f2 pendant DI
        t_seg = t[:signal_duration]
        seg_duration = T_emit
        k = (f2 - f1) / seg_duration
        t_rel = t_seg - t_seg[0]
        chirp_signal = np.sin(2 * np.pi * (f1 * t_rel + 0.5 * k * t_rel**2))
        signal_emit[:signal_duration] = chirp_signal

    else:
        raise ValueError("Type de modulation non valide. Choisissez parmi 'impulsionnel', 'FMCW', 'CW', 'chirp'.")

    # Signal reçu (identique à l'émetteur dans cet exemple)
    signal_received = np.zeros_like(t)
    signal_received[:signal_duration] = signal_emit[:signal_duration]

    # Sauvegarde en CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Signal Emitted", "Signal Received"])
        for i in range(len(t)):
            writer.writerow([t[i], signal_emit[i], signal_received[i]])

    return t, signal_emit, signal_received, T_emit, dtoa

##########################################
# Code pour détecter la zone DI et FFT   #
##########################################

# Exemple d'utilisation
if __name__ == "__main__":
    DI = [0.05, 0.1, 0.2]
    DTOA = [0.2, 0.3, 0.4]
    f1 = 100.0
    f2 = 300.0
    sampling_rate = 2e4
    modulation = 'chirp'  # Essayer 'impulsionnel', 'FMCW', 'CW', 'chirp'
    t, signal_emit, signal_received, T_emit, dtoa = generate_radar_signal(
        f1, f2, DI, DTOA, modulation, sampling_rate=sampling_rate, output_csv="radar_signal_example.csv"
    )

    # On sait que la zone DI est de 0 à T_emit
    signal_duration = int(T_emit * sampling_rate)

    # Extraire la portion DI du signal
    di_signal = signal_emit[:signal_duration]
    N = len(di_signal)

    # Réaliser la FFT sur la zone DI
    freqs = np.fft.fftfreq(N, d=1/sampling_rate)
    di_fft = np.fft.fft(di_signal)
    half = N//2
    freqs_plot = freqs[:half]
    fft_magnitude = np.abs(di_fft[:half])*(2/N)  # Normalisation

    # Affichage du signal DI et de son spectre
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(signal_duration)/sampling_rate, di_signal)
    plt.title("Signal DI")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.subplot(1,2,2)
    plt.plot(freqs_plot, fft_magnitude)
    plt.title("Spectre (FFT) de la zone DI")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude Normalisée")
    plt.tight_layout()
    plt.show()
