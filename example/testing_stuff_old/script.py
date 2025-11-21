filename = "LiH30_complex_pulse_penalty_on_amps.out"  
import numpy as np
# Initialize 6 empty lists
lists = [[] for _ in range(6)]
lists_error = [[] for _ in range(6)]
with open(filename, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "16-element Vector{Float64}" in line :
            try:
                for j in range(6):
                    value = float(lines[i + 1 + j].strip())
                    lists[j].append(value)
                i += 16  # Skip the 16 lines just read
            except Exception as e:
                print(f"Error parsing 16-element vector at line {i}: {e}")
        elif "6-element Vector{Float64}" in line :
            try:
                for j in range(6):
                    value = float(lines[i + 1 + j].strip())
                    lists_error[j].append(value)
                i += 6  # Skip the 6 lines just read
            except Exception as e:
                print(f"Error parsing 6-element vector at line {i}: {e}")
        i += 1


for i, lst in enumerate(lists_error):
    print(f"List {i+1}: {lst}")

#plotting MET vs error in energy
import matplotlib.pyplot as plt
T = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
plt.figure(figsize=(7, 6))
plt.plot(T, np.abs(lists_error[0]), 'o-', label='Ground state', color='blue')
plt.plot(T, np.abs(lists_error[1]), 's-', label='Excited state 1', color='red')
plt.plot(T, np.abs(lists_error[2]), 'd-', label='Excited state 2', color='green')
plt.plot(T, np.abs(lists_error[3]), 'v-', label='Excited state 3', color='purple')
plt.plot(T, np.abs(lists_error[4]), '^-', label='Excited state 4', color='orange')
plt.plot(T, np.abs(lists_error[5]), '*-', label='Excited state 5', color='brown')

plt.xlabel('MET')
plt.ylabel('Energy error (Hartree)')
plt.xticks(T, rotation=45)
# plt.grid(True)
# plt.tight_layout()
plt.show()