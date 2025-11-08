The program `train.py` uses the following training and validation data for temperature (in °C) and voltage (in V):

# Training data
| Temperature (°C) | Voltage (V) |
|------------------|-------------|
| 79.8             | 1.88        |
| 79.5             | 1.87        |
| 75.3             | 1.86        |
| 73.4             | 1.85        |
| 71.8             | 1.85        |
| 70.6             | 1.85        |
| 69.9             | 1.84        |
| 68.5             | 1.84        |
| 67.9             | 1.83        |
| 80.8             | 1.89        |
| 84.2             | 1.90        |
| 86.3             | 1.91        |
| 90.1             | 1.92        |
| 92.5             | 1.92        |
| 97.5             | 1.93        |
| 57.3             | 1.51        |
| 56.8             | 1.52        |
| 56.4             | 1.52        |
| 54.9             | 1.62        |
| 53.9             | 1.62        |
| 53.1             | 1.76        |
| 51.6             | 1.75        |
| 48.9             | 1.74        |

![Training Data](../figs/training.png)

# Validation data
| Temperature (°C) | Voltage (V) |
|------------------|-------------|
| 53.1             | 1.76        |
| 67.9             | 1.83        |
| 75.3             | 1.86        |
| 84.2             | 1.90        |
| 92.5             | 1.92        |
| 57.3             | 1.51        |

![Validation Data](../figs/validation.png)

It works by using the Callendar-Van Dusen equation to model the relationship between temperature and resistance, which is then converted to voltage using a voltage divider circuit. We have slightly modified it to directly use voltage readings for temperature estimation.