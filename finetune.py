import subprocess

# Define the command as a list of arguments
command = [
    "python", "scripts/training/train.py",
    "--config", "./scripts/training/configs/chronos-t5-tiny.yaml",
    "--model-id", "amazon/chronos-t5-small",
    "--no-random-init",
    "--max-steps", "1000",
    "--learning-rate", "0.001"
]

# Run the command without output buffering
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=True)

# Print the output line by line in real-time
for stdout_line in iter(process.stdout.readline, ""):
    print(stdout_line, end='')

# Capture the standard error
stderr = process.stderr.read()

# Close streams
process.stdout.close()
process.stderr.close()

# Ensure the subprocess ends properly
process.wait()

# Print the standard error if needed
print("Error:", stderr)
