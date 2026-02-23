# Vascular Geometry Estimation CNN

A machine learning project that trains convolutional neural networks to predict the **radius** and **circularity** of blood vessels in 15×15 pixel grayscale images.

## Project Overview

This project implements two independent CNN models that perform regression tasks:
- **Radius Model**: Predicts the radius of blood vessels
- **Circularity Model**: Predicts the circularity coefficient of blood vessels

Both models are trained on segmented image data and evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics.

## Key Features

✅ **Dual Model Architecture** - Separate models for radius and circularity  
✅ **Comprehensive Evaluation** - MAE, MSE, and visual analysis  
✅ **Error Quantification** - Detailed error analysis by target value bins  
✅ **Training Visualization** - Learning curves and convergence analysis  
✅ **Regularization** - Dropout layer to prevent overfitting
## Model Architecture

Both models use an identical 3-layer CNN architecture:

```
Input (15×15×1)
        ↓
Conv2D(8, kernel=(3,3), relu, padding='same')
        ↓
MaxPooling2D((2,2))
        ↓
Conv2D(16, kernel=(3,3), relu, padding='same')
        ↓
MaxPooling2D((2,2))
        ↓
Conv2D(32, kernel=(3,3), relu, padding='same')
        ↓
Flatten
        ↓
Dense(64, relu)
        ↓
Dropout(0.3)
        ↓
Dense(1, linear)  # Regression output
```

**Training Configuration:**
- Optimizer: Adam (learning_rate=1e-4)
- Loss Function: Mean Absolute Error (MAE)
- Metrics: MAE, MSE
- Epochs: 320
- Batch Size: 2
- Train/Val Split: 80/20

## Project Files

### Notebooks

- **`radius.ipynb`** - CNN model for radius prediction
  - Data loading and preprocessing
  - Model training and evaluation
  - Error analysis and visualization

- **`circularity.ipynb`** - CNN model for circularity prediction
  - Same structure as radius.ipynb
  - Predicts circularity coefficient instead of radius

## Dataset

The project uses segmented image data stored as NumPy arrays:

```
data/
├── inputs_segment0.npy    # Input images (15×15 grayscale)
├── inputs_segment1.npy
├── inputs_segment3.npy
├── inputs_segment4.npy
├── targets_segment0.npy   # Target values (radius, circularity)
├── targets_segment1.npy
├── targets_segment3.npy
└── targets_segment4.npy
```

**Data Characteristics:**
- Image shape: (N, 15, 15) - grayscale images
- Target shape: (N, 2) - [radius, circularity] for each image
- All segments are concatenated for training and validation
<img width="382" height="382" alt="obraz" src="https://github.com/user-attachments/assets/f2f7696a-be7b-4ae3-8e1e-66b88e439cd8" />
 
## Results Visualization

Each notebook generates:

1. **Learning Curves**
   - MAE and MSE for training and validation sets
   - Full training history (320 epochs)
   - Zoomed view of last 50 epochs
<img width="1389" height="790" alt="obraz" src="https://github.com/user-attachments/assets/d17950c1-194e-4a01-9009-d9721f0a716b" />

2. **Error Analysis**
   - Scatter plot: Prediction error vs. target value
   - Histogram: Distribution of target values with average error per bin
   - Error distribution with mean error line
<img width="600" height="450" alt="obraz" src="https://github.com/user-attachments/assets/dea85efd-8225-4b85-ad06-d08592a058d3" />
<img width="600" height="450" alt="obraz" src="https://github.com/user-attachments/assets/5774dde6-44f0-4b1d-b6a0-8edf86427abd" />

3. **Training Summary**
   - Accuracy metrics based on MAE
   - Overfitting/Underfitting detection
   - Final validation MAE
# Dependencies

```
numpy >= 1.19.0
matplotlib >= 3.0.0
tensorflow >= 2.8.0
scikit-learn >= 0.24.0
scipy >= 1.5.0
```

## Installation & Usage

### 1. Install Dependencies

```bash
pip install numpy matplotlib tensorflow scikit-learn scipy
```

### 2. Run Notebooks

```bash
# For radius prediction
jupyter notebook radius.ipynb

# For circularity prediction
jupyter notebook circularity.ipynb
```

### 3. Execute Cells

- Run all cells sequentially from top to bottom
- Data files must be in `data/` subdirectory
- Ensure sufficient disk space for model training (~1-2 GB)


## Author

- Patryk Pełka
- Robert Wewersowicz

## License

This project is provided for educational purposes.
