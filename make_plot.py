import matplotlib.pyplot as plt

# Load data from the text file
with open('./results/results.txt', 'r') as file:
    # Assuming each line contains one numeric value
    data = [float(line.strip()) for line in file]

# Separate values into two lists
greater_than_10 = [value for value in data if value > 10]
less_than_or_equal_10 = [value for value in data if value <= 10]

# Save the filtered values to separate text files
with open('./results/greater_than_10.txt', 'w') as gt10_file:
    for value in greater_than_10:
        gt10_file.write(f"{value}\n")

with open('./results/less_than_or_equal_10.txt', 'w') as le10_file:
    for value in less_than_or_equal_10:
        le10_file.write(f"{value}\n")

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot values greater than 10
ax1.plot(greater_than_10)
ax1.set_title('PSNR')
ax1.set_xlabel('Index')
ax1.set_ylabel('PSRN')
ax1.set_ylim(0, max(greater_than_10) if greater_than_10 else 1)  # Start Y-axis from 0
ax1.grid(True)

# Plot values less than or equal to 10
ax2.plot(less_than_or_equal_10, marker='o', linestyle='-')
ax2.set_title('SS')
ax2.set_xlabel('Index')
ax2.set_ylim(0, max(less_than_or_equal_10) if less_than_or_equal_10 else 1)  # Start Y-axis from 0
ax2.grid(True)

# Display the plots
plt.tight_layout()
plt.show()
