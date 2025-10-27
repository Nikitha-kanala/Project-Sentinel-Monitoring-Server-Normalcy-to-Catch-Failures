import numpy as np # Import numerical processing library
import matplotlib.pyplot as plt # Import plotting library
from sklearn.svm import OneClassSVM # Import the anomaly detection model

print("Libraries imported.")

# --- 1. Simulate "Real-Time" Server Data ---

# Create "normal" server behavior:
# Let's say 'CPU Usage' and 'Memory Usage' are correlated.
# We'll generate 200 "normal" data points.
np.random.seed(42) # Set a random seed for reproducible results
X_normal = np.random.randn(200, 2) * 0.5 + [50, 60]  # Create 200 normal (CPU, Mem) data points

# Create "anomalous" server behavior:
# These are the "failures" our model has NEVER seen.
X_anomalies = np.array([
    [40, 95],  # Define a memory leak anomaly
    [85, 88],  # Define an overload anomaly
    [30, 30]   # Define a crash anomaly
])

# Combine them into one "live stream"
X_full_stream = np.vstack([X_normal, X_anomalies]) # Stack normal and anomaly data together

print(f"Simulated {len(X_normal)} normal data points.")
print(f"Injected {len(X_anomalies)} anomalies into the stream.")

# --- 2. Train the "Wow" Model (One-Class SVM) ---

# This is the most important step.
# We create the model. 'nu' (nu) is the key parameter.
# It's our guess of the % of anomalies in the data (e.g., 0.05 = 5%)
# 'kernel' and 'gamma' help it find a complex, non-linear boundary.
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05) # Initialize the One-Class SVM model

# *** THE KEY INSIGHT ***
# We train the model ONLY on the "normal" data.
# It is learning the "boundary of normalcy."
print("Training model on NORMAL data only...")
model.fit(X_normal) # Train the model *only* on the normal data

# --- 3. Run the "Real-Time" Detection ---

# Now, we feed the *entire* data stream (normal + anomalies) to the model.
# The model will output:
#  +1 if it's "normal" (an inlier)
#  -1 if it's "anomalous" (an outlier)
print("Detecting anomalies in the full data stream...")
predictions = model.predict(X_full_stream) # Get predictions for the *entire* dataset

# Get the indices of the anomalies our model found
anomaly_indices = np.where(predictions == -1)[0] # Find the indices of data points flagged as anomalies (-1)
found_anomalies = X_full_stream[anomaly_indices] # Get the actual (CPU, Mem) values of the found anomalies

print(f"Model successfully identified {len(found_anomalies)} anomalies!")

# --- 4. Visualize the "Astonishing" Result ---
print("Generating plot...")

# Create a meshgrid to visualize the "boundary of normalcy"
xx, yy = np.meshgrid(np.linspace(20, 90, 500), np.linspace(20, 100, 500)) # Create a grid for the plot background
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]) # Get the model's decision boundary score for each grid point
Z = Z.reshape(xx.shape) # Reshape the scores to match the grid

plt.figure(figsize=(10, 6)) # Set the size of the plot

# Plot the "normalcy" boundary
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palegreen', alpha=0.4, label='Learned "Normalcy"') # Fill the "normal" area

# Plot all data points
plt.scatter(X_full_stream[:, 0], X_full_stream[:, 1], c='gray', label='All Server Pings') # Plot all data points in gray

# Highlight the *real* anomalies in Red
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='red', s=100, label='True Anomalies (Failures)') # Highlight the *actual* anomalies

# Circle the anomalies *our model found*
plt.scatter(found_anomalies[:, 0], found_anomalies[:, 1], 
            facecolors='none', edgecolors='red', s=150, 
            linewidths=2, label='Model-Detected Anomalies') # Circle the anomalies the model *found*

plt.title('Proactive Server Failure Detection (One-Class SVM)', fontsize=14) # Set the plot title
plt.xlabel('CPU Usage (%)') # Set the x-axis label
plt.ylabel('Memory Usage (%)') # Set the y-axis label
plt.legend() # Show the plot legend
plt.show() # Display the final plot