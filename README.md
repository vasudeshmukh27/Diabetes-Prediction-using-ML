# 🩺 Diabetes Prediction using Machine Learning

This project predicts whether a person has diabetes or not using **machine learning models**.  
We use the **Pima Indians Diabetes Dataset**, perform exploratory data analysis (EDA), train multiple classifiers, and compare their performance.  

---

## 📌 Project Workflow

1. **Data Exploration**
   - Check dataset dimensions, column info, and class distribution.
   - Handle imbalance by visualization (`seaborn countplot`).
   - Inspect dataset statistics.

2. **Data Visualization**
   - Class distribution plots
   - Correlation heatmap
   - Feature importance plots

3. **Model Training**
   - Train/Test Split (Stratified to maintain class balance).
   - Models used:
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree Classifier** (with and without pre-pruning).

4. **Model Evaluation**
   - Compare training and testing accuracies.
   - Detect overfitting in Decision Trees and fix using `max_depth`.
   - Visualize feature importances.

---

## ⚙️ Technologies Used

- **Python 3**
- **NumPy, Pandas** → Data handling
- **Matplotlib, Seaborn** → Visualization
- **Scikit-learn** → ML models & evaluation

---

## 📊 Results

### K-Nearest Neighbors (KNN)
- Explored values of *k* from 1 to 10.
- Optimal value: **k = 9**
- Accuracy:
  - Training: ~85%
  - Testing: ~77%

### Decision Tree
- Without pruning:
  - Training Accuracy: **100%**
  - Testing Accuracy: **much lower** → Overfitting  
- With pre-pruning (`max_depth=3`):
  - Training Accuracy: ~80%
  - Testing Accuracy: ~78% → Better generalization  

---

## 🌟 Key Insights
- **Overfitting** can be reduced by applying pre-pruning (`max_depth`).
- Some features contribute more strongly to predictions than others (via feature importance).
- Simple models like **KNN with tuned hyperparameters** perform comparably to more complex models.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vasudeshmukh27/Diabetes-Prediction-using-ML.git diabetes-prediction
   cd diabetes-prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook diabetes_prediction.ipynb

---

## 🙌 Acknowledgements

Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
Inspired by real-world healthcare applications of ML.

---

## 📧 Contact

Created with ❤️ by Vasu Deshmukh
If you found this useful, feel free to ⭐ star the repo and connect with me on [LinkedIn](https://www.linkedin.com/in/vasu-deshmukh/).
