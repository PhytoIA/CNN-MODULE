import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Sample data for soil moisture, temperature, and crop performance
soil_moisture = np.array([30, 35, 32, 45, 40])  # percentage
temperature = np.array([18, 21, 19, 23, 22])    # Celsius
crop_yield = np.array([80, 85, 83, 90, 88])     # yield per hectare


# Labels for optimal irrigation and fertilization in percentage
irrigation = np.array([20, 25, 22, 30, 28])   # water in percentage
fertilizer = np.array([5, 6, 5, 7, 6])        # fertilizer in kg/ha


# Train a model for irrigation schedule
irrigation_model = RandomForestRegressor()
irrigation_model.fit(np.column_stack((soil_moisture, temperature, crop_yield)), irrigation)


# Train a model for fertilizer schedule
fertilizer_model = RandomForestRegressor()
fertilizer_model.fit(np.column_stack((soil_moisture, temperature, crop_yield)), fertilizer)


# Simulating new data for a prediction
new_soil_moisture = 38
new_temperature = 20
new_crop_yield = 85


predicted_irrigation = irrigation_model.predict([[new_soil_moisture, new_temperature, new_crop_yield]])
predicted_fertilizer = fertilizer_model.predict([[new_soil_moisture, new_temperature, new_crop_yield]])


print(f"Predicted irrigation schedule: {predicted_irrigation[0]:.2f}% water")
print(f"Predicted fertilizer plan: {predicted_fertilizer[0]:.2f} kg/ha")
