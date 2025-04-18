# Loan Default Prediction and Segmentation

This project focuses on building a classification model to predict whether a borrower will default on a loan using historical financial and credit score data. In addition to classification, it also applies clustering techniques to uncover hidden patterns in the dataset through customer segmentation.

## 📌 Project Highlights

- Classification using Random Forest to predict loan defaults
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix Heatmap
- Clustering with KMeans for segmentation
- Dimensionality reduction using PCA for visualization

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Google Colab (for execution)

---

## 📁 Dataset

You’ll need a dataset in CSV format with features like:
- Financial history
- Credit score
- User behavior
- A target column like `Default` or similar (1 = default, 0 = no default)

Upload the dataset manually when prompted in the Colab notebook.

---

## 🚀 How to Run

1. Open the project in **Google Colab**
2. Upload your dataset when prompted
3. Run the code cells one by one

The notebook will:
- Clean and preprocess your data
- Train a Random Forest classifier
- Generate a heatmap of the confusion matrix
- Print evaluation metrics
- Perform clustering using KMeans
- Visualize clusters in a 2D PCA plot

---

## 📊 Output Examples

- Confusion Matrix Heatmap
- Accuracy, Precision, Recall scores
- KMeans Cluster Visualization using PCA

---

## 💡 Notes

- You can adjust the model or number of clusters based on your dataset
- PCA is used for visualization only, not for training the classifier

---

## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or fork the repository.

