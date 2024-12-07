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
    - 'FMCW' : Variation de fréquence continue, rampe triangulaire (f1 -> f2 -> f1) sur tout DI.
    - 'CW' : Une seule fréquence constante pendant DI.
    - 'chirp' : Un seul chirp linéaire f1->f2 pendant DI.

    Paramètres :
    - f1, f2 : fréquences min et max (Hz)
    - DI_list : liste des durées d'émission possibles (s)
    - DTOA_list : liste des DTOA possibles (s)
    - modulation_type : 'impulsionnel', 'FMCW', 'CW', 'chirp'
    - sampling_rate : fréquence d'échantillonnage (Hz)
    - output_csv : nom du fichier CSV de sortie

    Le signal est non nul uniquement pendant DI, puis nul jusqu'à DTOA.
    """

    # Choix aléatoire de DI et DTOA
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
        # Centre l'impulsion au milieu du DI
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
                # Phase montée
                f_inst[i] = f1 + (f2 - f1)*(tau/half)
            else:
                # Phase descente
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
        # Chirp : un seul chirp linéaire f1->f2 pendant tout DI
        t_seg = t[:signal_duration]
        seg_duration = T_emit
        k = (f2 - f1) / seg_duration
        t_rel = t_seg - t_seg[0]
        chirp_signal = np.sin(2 * np.pi * (f1 * t_rel + 0.5 * k * t_rel**2))
        signal_emit[:signal_duration] = chirp_signal

    else:
        raise ValueError("Type de modulation non valide. Choisissez parmi 'impulsionnel', 'FMCW', 'CW', 'chirp'.")

    # Signal reçu (identique à l'émis)
    signal_received = np.zeros_like(t)
    signal_received[:signal_duration] = signal_emit[:signal_duration]

    # Sauvegarde en CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Signal Emitted", "Signal Received"])
        for i in range(len(t)):
            writer.writerow([t[i], signal_emit[i], signal_received[i]])

    print(f"Signal data saved to {output_csv}")

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_emit, label="Signal Emitted", alpha=0.7)
    plt.plot(t, signal_received, label=f"Signal Received (DTOA = {dtoa:.4f} s)", alpha=0.7, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Radar Signal - {modulation_type.upper()} | DI={T_emit:.3f}s, DTOA={dtoa:.3f}s")
    plt.legend()
    plt.grid(True)
    plt.show()


##################################
# Tests pour chaque modulation   #
##################################

# Paramètres communs
DI = [0.05, 0.1, 0.2]   # Durées possibles
DTOA = [0.2, 0.3, 0.4]  # DTOA possibles
f1 = 100.0
f2 = 300.0
sampling_rate = 2e4

# Test Impulsionnel (une seule impulsion)
generate_radar_signal(
    f1, f2,
    DI, DTOA,
    modulation_type='impulsionnel',
    sampling_rate=sampling_rate,
    output_csv="test_impulsionnel.csv"
)

# Test FMCW (rampe triangulaire)
generate_radar_signal(
    f1, f2,
    DI, DTOA,
    modulation_type='FMCW',
    sampling_rate=sampling_rate,
    output_csv="test_fmcw.csv"
)

# Test CW (une seule porteuse constante)
generate_radar_signal(
    f1, f2,
    DI, DTOA,
    modulation_type='CW',
    sampling_rate=sampling_rate,
    output_csv="test_cw.csv"
)

# Test Chirp (un seul chirp linéaire)
generate_radar_signal(
    f1, f2,
    DI, DTOA,
    modulation_type='chirp',
    sampling_rate=sampling_rate,
    output_csv="test_chirp.csv"
)
