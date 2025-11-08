# Example data: Voltage (V) and corresponding Temperature (T)
# You can replace these with your own measured data
V = np.array([1.755,  1.764,  1.769, 1.799, 1.711, 1.720,  1.735, 1.740,  1.769,  1.764, 1.760, 1.862, 1.857, 1.852,  1.848,  1.843,  1.838,  1.833,  1.828,  1.823,  1.818,  1.81,  1.804])
T = np.array([40.1,    40.5,  40.7,   47.4,  28.2,  28.7,   38,    33,     34.6,   37.8,  37.1,  57.8,  57,    56.6,   55.5,   55.2,   54.5,   53.9,   53.5,   50.2,   48.7,   44.8,  44.3])

# Convert to column vectors
V = V.reshape(-1, 1)
T = T.reshape(-1, 1)

# Build the design matrix X = [1, V, V²]
X = np.hstack((np.ones_like(V), V, V**2))

# Solve least squares:  X * n = T  --> n = [a, b, c]^T
a, b, c = np.linalg.lstsq(X, T, rcond=None)[0]

# Display the results
print("Calculated coefficients:")
print(f"a = {a[0]:.6f}")
print(f"b = {b[0]:.6f}")
print(f"c = {c[0]:.6f}")


# Print the fitted equation
print(f"\nEquation: T = {a[0]:.6f} + ({b[0]:.6f})*V + ({c[0]:.6f})*V²")
