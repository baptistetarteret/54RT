import numpy as np
import matplotlib.pyplot as plt
import csv

def generate_radar_signal(
    f1, f2, 
    DI_list, DTOA_list, 
    modulation_type, 
    sampling_rate=1e5
):
    """
    Génère un signal radar selon le type de modulation spécifié.
    Modulations disponibles :
    - 'impulsionnel'
    - 'FMCW'
    - 'CW'
    - 'chirp'

    Retourne:
    t, signal_emit, signal_received, T_emit, dtoa, f_random (ou None)
    """
    idx = np.random.randint(0, len(DI_list))
    T_emit = DI_list[idx]
    dtoa = DTOA_list[idx]

    T_total = dtoa
    t = np.arange(0, T_total, 1 / sampling_rate)

    signal_emit = np.zeros_like(t)
    signal_duration = int(T_emit * sampling_rate)

    # Variable f_random : On la met à None si non utilisée
    f_random = None

    if modulation_type == 'impulsionnel':
        f_random = np.random.uniform(f1, f2)
        t_imp = t[:signal_duration]
        seg_duration = T_emit
        t_centered = t_imp - (seg_duration / 2)
        sigma = T_emit / 6.0
        envelope = np.exp(-t_centered**2 / (2 * sigma**2))
        impulse_signal = envelope * np.sin(2 * np.pi * f_random * t_imp)
        signal_emit[:signal_duration] = impulse_signal

    elif modulation_type == 'FMCW':
        # FMCW continu : pas de f_random unique, juste une rampe f1->f2->f1
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
        # CW : une seule fréquence constante
        f_random = np.random.uniform(f1, f2)
        t_seg = t[:signal_duration]
        cw_signal = np.sin(2 * np.pi * f_random * t_seg)
        signal_emit[:signal_duration] = cw_signal

    elif modulation_type == 'chirp':
        # Chirp linéaire f1->f2
        t_seg = t[:signal_duration]
        seg_duration = T_emit
        k = (f2 - f1) / seg_duration
        t_rel = t_seg - t_seg[0]
        chirp_signal = np.sin(2 * np.pi * (f1 * t_rel + 0.5 * k * t_rel**2))
        signal_emit[:signal_duration] = chirp_signal

    else:
        raise ValueError("Type de modulation non valide. Choisissez parmi 'impulsionnel', 'FMCW', 'CW', 'chirp'.")

    signal_received = np.zeros_like(t)
    signal_received[:signal_duration] = signal_emit[:signal_duration]

    return t, signal_emit, signal_received, T_emit, dtoa, f_random


def generate_and_save_all(
    f1, f2, 
    DI_list, DTOA_list, 
    modulation_type, 
    sampling_rate=1e5, 
    output_csv="result_all.csv"
):
    # Génération du signal
    t, signal_emit, signal_received, T_emit, dtoa, f_random = generate_radar_signal(
        f1, f2, DI_list, DTOA_list, modulation_type, sampling_rate
    )

    # Extraction de la portion DI
    signal_duration = int(T_emit * sampling_rate)
    di_signal = signal_emit[:signal_duration]
    N = len(di_signal)

    # FFT sur la zone DI
    freqs = np.fft.fftfreq(N, d=1/sampling_rate)
    di_fft = np.fft.fft(di_signal)
    half = N // 2
    freqs_plot = freqs[:half]
    fft_magnitude = np.abs(di_fft[:half])*(2/N)  # Normalisation amplitude

    # Sauvegarde dans un CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ecrire les paramètres d'entrée
        writer.writerow(["f1", f1])
        writer.writerow(["f2", f2])
        writer.writerow(["DI_list"] + DI_list)
        writer.writerow(["DTOA_list"] + DTOA_list)
        writer.writerow(["modulation_type", modulation_type])
        writer.writerow(["sampling_rate", sampling_rate])
        writer.writerow(["Chosen_T_emit", T_emit])
        writer.writerow(["Chosen_dtoa", dtoa])
        # Ajouter f_random si disponible
        if f_random is not None:
            writer.writerow(["Chosen_f_random", f_random])
        else:
            writer.writerow(["Chosen_f_random", "N/A"])

        # Séparateur pour la partie Time Domain
        writer.writerow(["TIME_DOMAIN_DATA"])
        writer.writerow(["Time (s)", "Signal Emitted", "Signal Received"])
        for i in range(len(t)):
            writer.writerow([t[i], signal_emit[i], signal_received[i]])

        # Séparateur pour la partie Frequency Domain
        writer.writerow(["FREQUENCY_DOMAIN_DATA"])
        writer.writerow(["Frequency (Hz)", "Amplitude"])
        for i in range(len(freqs_plot)):
            writer.writerow([freqs_plot[i], fft_magnitude[i]])

    print(f"All data saved to {output_csv}")

    # Affichage graphique (optionnel)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(signal_duration)/sampling_rate, di_signal)
    plt.title("Signal DI")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.subplot(1,2,2)
    plt.plot(freqs_plot, fft_magnitude)
    plt.title("FFT DI")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude Normalisée")
    plt.tight_layout()
    plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    DI = [0.05, 0.1, 0.2]
    DTOA = [0.2, 0.3, 0.4]
    f1 = 100.0
    f2 = 300.0
    sampling_rate = 2e4
    modulation_type = 'chirp'  # test avec chirp
    
    generate_and_save_all(
        f1, f2, 
        DI, DTOA, 
        modulation_type, 
        sampling_rate=sampling_rate, 
        output_csv="all_data_example.csv"
    )
