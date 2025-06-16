# Simulate training and validation loss curves over epochs (mimicking large dataset behavior)
epochs = np.arange(1, 21)

# Simulated loss for each model
train_rf = 0.6 * np.exp(-0.2 * epochs) + 0.03 * np.random.rand(len(epochs))
val_rf = 0.65 * np.exp(-0.18 * epochs) + 0.04 * np.random.rand(len(epochs))

train_xgb = 0.55 * np.exp(-0.22 * epochs) + 0.02 * np.random.rand(len(epochs))
val_xgb = 0.6 * np.exp(-0.2 * epochs) + 0.03 * np.random.rand(len(epochs))

train_knn = 0.7 * np.exp(-0.15 * epochs) + 0.05 * np.random.rand(len(epochs))
val_knn = 0.75 * np.exp(-0.13 * epochs) + 0.06 * np.random.rand(len(epochs))

train_hwe = 0.5 * np.exp(-0.25 * epochs) + 0.02 * np.random.rand(len(epochs))
val_hwe = 0.55 * np.exp(-0.23 * epochs) + 0.03 * np.random.rand(len(epochs))

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_rf, label='Train - Random Forest')
plt.plot(epochs, val_rf, linestyle='--', label='Validation - Random Forest')

plt.plot(epochs, train_xgb, label='Train - XGBoost')
plt.plot(epochs, val_xgb, linestyle='--', label='Validation - XGBoost')

plt.plot(epochs, train_knn, label='Train - KNN')
plt.plot(epochs, val_knn, linestyle='--', label='Validation - KNN')

plt.plot(epochs, train_hwe, label='Train - HWE–PM')
plt.plot(epochs, val_hwe, linestyle='--', label='Validation - HWE–PM')

plt.title('Simulated Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
