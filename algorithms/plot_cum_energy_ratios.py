
import matplotlib.pyplot as plt

def plot_cumulative_energy(cum_energy_r, cum_energy_g, cum_energy_b, energy_ratio, num_sv_red, num_sv_green, num_sv_blue, S_r):
    # Plot cumulative energy ratios for each channel
    plt.figure(figsize=(8, 6))
    plt.plot(cum_energy_r, 'r', label='Red channel')
    plt.plot(cum_energy_g, 'g', label='Green channel')
    plt.plot(cum_energy_b, 'b', label='Blue channel')
    plt.plot([0, len(S_r)], [energy_ratio, energy_ratio], 'k--', label='Desired energy ratio')
    plt.plot([num_sv_red, num_sv_red], [0, 1], 'r--', label='Best number of singular values (red channel)')
    plt.plot([num_sv_green, num_sv_green], [0, 1], 'g--', label='Best number of singular values (green channel)')
    plt.plot([num_sv_blue, num_sv_blue], [0, 1], 'b--', label='Best number of singular values (blue channel)')
    plt.xlabel('Number of singular values')
    plt.ylabel('Cumulative energy ratio')
    plt.title('Cumulative energy ratios for each color channel')
    plt.legend()
    plt.show()