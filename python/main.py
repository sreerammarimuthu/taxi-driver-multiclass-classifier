# main.py

import sys
import torch
from torch.utils.data import DataLoader
from extract_feature import load_data
from sklearn.model_selection import train_test_split
from train import train_model, CustomDataset
from test import test_model

def main():
  if len(sys.argv) != 2:
    print("Usage: python main.py <mode>")
    print("<mode> should be 'train' or 'test'")
    sys.exit(1)
    
  mode = sys.argv[1]

  train_val_folder_path = '/content/drive/MyDrive/Colab Notebooks/BDA_Project 2/data_5drivers' # Path for Training Data 
  X_train_val, y_train_val = load_data(train_val_folder_path)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                    y_train_val,
                                                    test_size=0.2,
                                                    random_state=42)

  train_dataset = CustomDataset(X_train, y_train)
  val_dataset = CustomDataset(X_val, y_val)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  input_dim = 8
  hidden_dim = 32
  output_dim = 5

  if mode == 'train':
    print("Starting training process...")
    train_model(train_loader, val_loader, input_dim, hidden_dim,
                output_dim)
  elif mode == 'test':
    print("Starting testing process...")
    test_folder_path = '/content/drive/MyDrive/Colab Notebooks/BDA_Project 2/Test' # Path for Test Data
    X_test, y_test = load_data(test_folder_path)
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_model(test_loader, input_dim, hidden_dim,
               output_dim)
  else:
    print("Invalid mode. Choose 'train' or 'test'.")
    sys.exit(1)


if __name__ == '__main__':
  main()
