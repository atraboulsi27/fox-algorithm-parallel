import matplotlib.pyplot as plt

# Given number processes and their execution times
numbers = [16, 25, 36, 49, 64, 81, 100, 121, 144]
execution_times = [0.41, 0.51, 0.59, 1.1, 1.4, 1.48, 1.42, 2.03, 2.87]

# Plotting the data
plt.figure(figsize=(8, 6))
plt.plot(numbers, execution_times, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Number Processes')
plt.ylabel('Execution Time (Seconds)')
plt.title('Execution Time vs. Number Processes')

# Show the exact numbers on the y axis
plt.yticks(execution_times, [f'{time:.2f}' for time in execution_times])

# Display the plot
plt.grid(True)
plt.show()
