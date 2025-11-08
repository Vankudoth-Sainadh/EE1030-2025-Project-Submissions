import matplotlib.pyplot as plt

temperature = [32.1, 36.6, 41.1, 45.5, 50.8, 56.8, 62.1, 67.6,71.8,76.6,81.3,85.9,90.9,95.9,97.6]
voltage = [2.66, 2.67,2.68,2.69,2.71,2.74,2.75,2.77,2.79,2.81,2.82,2.83,2.85,2.87,2.89]


fig, ax = plt.subplots(figsize=(8, 6))

line_color = '#E0D450'
marker_face_color = '#F0E68C'

ax.plot(
    temperature,
    voltage,
    color=line_color,
    linewidth=2.5,
    marker='o',
    markersize=9,
    markerfacecolor=marker_face_color,
    markeredgecolor=line_color,
    markeredgewidth=1.5
)

ax.set_title('Temperature vs. Voltage', fontsize=16, pad=20, color='black')
ax.set_xlabel('Temperature (°C)', fontsize=12, labelpad=15, color='black')
ax.set_ylabel('Voltage (V)', fontsize=12, labelpad=15, color='black')

ax.grid(True, color='#CCCCCC', linestyle='-', linewidth=1)

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

plt.tight_layout()

plt.show()

