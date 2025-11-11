# Prerequisites :
1. Python >= 3.8
2. PyTorch >= 1.9
3. PyTorchLightning >=1.9.0
4. Hydra >=1.1.0
5. ASE >=3.21
6. Install SchNet https://github.com/atomistic-machine-learning/schnetpack

# Prepare input
input_npz.py file to save input in .npz format.
npz_to_db.py to transform .npz to .db format.

# Run 
python3 training_module.py &

# Generated output files
After successful training, generates two output files - actual vs predicted output for training data and testing data. The converged model will be stored in *best_inference_model* file.

