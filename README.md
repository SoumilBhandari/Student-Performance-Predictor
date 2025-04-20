# Student Exam Score Predictor & Recommender

This repository trains a Random Forest regression model on student habits data to predict exam scores and provides personalized recommendations:

- **Predict** exam score from features such as study hours, sleep, attendance, etc.
- **Classify** students as *at risk* (predicted score < 50) or *on track*.
- **Recommend** remedial actions for at-risk students and optimization tips for on-track students.

## Repository Structure

```
student_performance_predictor_repo/
├── LICENSE
├── README.md
├── advanced_predict_optimize.py
├── .gitignore
└── data/
    └── student_habits_performance.csv
```

## Setup & Usage

1. **Clone** or download the repository.
2. **Install** dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. **Run**:
   ```bash
   python advanced_predict_optimize.py
   ```

The script will output test metrics and a sample student recommendation.

## License

This project is licensed under the MIT License.
