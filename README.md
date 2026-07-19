# Traffic Sign Recognition using CNN

A deep learning project that classifies German traffic signs into 43 categories
using a Convolutional Neural Network (CNN), trained on the GTSRB (German Traffic
Sign Recognition Benchmark) dataset — a classic building block for autonomous
vehicle perception systems.

## What's in this notebook

A single, self-contained Colab notebook covering:

- **Data loading & preprocessing** — resizing all images to 32x32, normalizing pixel values
- **Exploratory Data Analysis** — class distribution across all 43 sign types (naturally imbalanced)
- **CNN architecture** — two convolutional blocks (Conv2D + BatchNorm + MaxPooling + Dropout) followed by a dense classification head
- **Training** — Adam optimizer, sparse categorical cross-entropy loss, early stopping on validation loss
- **Evaluation** — accuracy/loss curves, confusion matrix, sample predictions with true vs. predicted labels
- **Classical ML baselines** — KNN, Decision Tree, and Random Forest trained on PCA-reduced pixel features, for comparison against the CNN
- **Unsupervised clustering** — K-Means to explore whether traffic sign classes naturally separate in pixel-feature space without using labels
- **RNN (LSTM)** — an experimental approach treating each image as a sequence of pixel rows, compared against the CNN
- **NLP** — text-based semantic grouping of the traffic sign label names themselves (e.g., clustering all "Speed limit" signs together) using TF-IDF

## Why a CNN?

Traffic sign images require learning spatial visual patterns (shapes, colors, symbols)
directly from pixels. CNNs are purpose-built for this, using convolutional filters
that automatically learn these features — unlike classical ML models, which need
manually engineered or flattened pixel inputs and generally perform worse on raw
image data, as shown in the model comparison section.

## How to run it

1. Open the notebook in [Google Colab](https://colab.research.google.com)
2. Run the setup cell to install dependencies
3. Download the dataset from Kaggle:
   [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
   and upload/extract it so `Train/`, `Train.csv` are present
4. Run all remaining cells in order

## Dataset

GTSRB - German Traffic Sign Recognition Benchmark, via Kaggle:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Reference: Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. *Man vs. Computer:
Benchmarking Machine Learning Algorithms for Traffic Sign Recognition.* Neural Networks, 2012.

## Key Techniques Demonstrated

`Python` `TensorFlow/Keras` `CNN` `RNN/LSTM` `Scikit-learn` `KNN` `Decision Tree`
`Random Forest` `K-Means Clustering` `PCA` `NLP/TF-IDF` `Multi-class Image Classification`
