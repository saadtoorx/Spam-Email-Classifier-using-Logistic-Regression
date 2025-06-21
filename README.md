# 📧 Spam Email Classifier using Logistic Regression

Detect spam messages using machine learning and text analysis with a logistic regression model.

---

## 🧠 About the Project

This beginner-friendly machine learning project demonstrates how to build a binary classification model to distinguish between spam and non-spam (ham) emails. The model is trained using the **TF-IDF vectorization** technique to convert text data into numerical features, followed by a **Logistic Regression** algorithm from `scikit-learn`. The script also includes an interactive prediction feature where users can input their own email text and get instant classification results.

---

## 🚀 Features

* 📩 Preprocess and clean labeled email dataset
* ✒️ Convert text data into numerical vectors using TF-IDF
* 📊 Train a Logistic Regression classifier
* 📈 Evaluate performance using Accuracy, Precision, Recall, and F1 Score
* 🔍 Visualize confusion matrix with `seaborn`
* 🧠 Predict live email content using user input

---

## 🛠️ Tech Stack

* Python 3.x  
* pandas  
* scikit-learn  
* seaborn  
* matplotlib  

---

## 📁 Project Structure

```
spam-email-classifier/
├── main.py                  # Core script for training and prediction
├── email.csv                # Dataset (labeled spam and ham emails)
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── requirements.txt         # Required Python libraries
```

---

## 💻 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/saadtoorx/spam-email-classifier.git
cd spam-email-classifier
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Script**

```bash
python main.py
```

4. **Classify Your Own Email Content**

When prompted, type your email message to get a prediction:

```text
📧 Type your email content below (type 'quit' to exit):

Your Email: Congratulations! You won a free vacation.
🧠 Prediction: SPAM

Your Email: Please review the attached report before tomorrow's meeting.
🧠 Prediction: HAM (Not Spam)
```

---

## 📊 Sample Output

```
Accuracy Score: 0.9832
Precision: 0.9625
Recall: 0.9573
F1 Score: 0.9599
```

Includes a confusion matrix heatmap plotted with `seaborn`.

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Made with ❤️ by [@saadtoorx](https://github.com/saadtoorx)

If this project helped you or inspired you, don’t forget to ⭐ the repo and share your feedback!
