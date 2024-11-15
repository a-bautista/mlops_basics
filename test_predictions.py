import requests

# Define the API URL
url = "http://localhost:8000/predict"

# List of all model choices to test
model_choices = ["log_smoteenn", "log_tomeklinks", "log_randover", "log_smote", "log_default"]

# Test samples with expected outputs (last column is the Diabetes_binary label)
test_samples = [
    {
        "features": [1, 0, 1, 30, 1, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 14, 0, 0, 9, 6, 7],
        "expected_label": 0
    },
    {
        "features": [1, 1, 1, 25, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 11, 4, 4],
        "expected_label": 0
    },
    {
        "features": [1, 1, 1, 30, 1, 0, 1, 0, 1, 1, 0, 1, 0, 5, 30, 30, 1, 0, 9, 5, 1],
        "expected_label": 1
    }
]

# Iterate over each model and test each sample
for model_name in model_choices:
    print(f"\nTesting model: {model_name}")
    
    for i, sample in enumerate(test_samples, start=1):
        # Prepare the request payload with the current model choice
        payload = {
            "features": sample["features"],
            "model_choice": model_name
        }
        
        # Make the API request
        response = requests.post(url, json=payload)
        result = response.json()
        
        # Extract prediction from API response
        predicted_label = result["prediction"]
        expected_label = sample["expected_label"]
        
        # Output the result for each test sample and model
        print(f"  Test Sample {i}:")
        print(f"    Expected: {expected_label}, Predicted: {predicted_label}")
        print("    Result:", "Match" if predicted_label == expected_label else "Mismatch")
    print("\n" + "-" * 50)
