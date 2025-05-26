# Taxi Driver Multiclass Classifier

This project applies deep learning techniques to classify taxi drivers based on their daily driving routes. Using real-world GPS data with features like latitude, longitude, timestamp, and status (occupied/vacant), we extract 100-step sub-trajectories for each driver and train an LSTM-based sequence classifier to predict the driver identity. 

The model helps us uncover driving behavior patterns with applications in: Driver monitoring, Anomaly detection, and Route optimization.

## Contents  

`data/`  
- `dataset.md` - Description of trajectory dataset source, format and features.

`models/`  
- `best_model.pt` & `best_model.pth` - Saved model checkpoints from training.

`python/`  
- `extract_feature.py` - Data loading, preprocessing, feature engineering, and sub-trajectory extraction
- `model.py` - LSTM model architecture for sequence classification
- `train.py` - Full training + validation loop with checkpointing and metrics
- `test.py` - Evaluation script to test the model on new data
- `main.py` - Integrated entry point for training or testing
- `howtorun.md` - Example run commands 

## Results  

Best Model: LSTM with tuned learning rate, trained for 40 epochs
Accuracy improved consistently with epochs as expected. 
LSTM outperformed baseline RNN in classification accuracy. Model captured temporal-spatial driver patterns effectively without overfitting. 
