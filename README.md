# Customer Churn Prediction Dashboard

A professional machine learning-powered web application for predicting customer churn risk using advanced classification models. Built with Streamlit and scikit-learn.

## 🎯 Overview

This dashboard uses a trained machine learning model to identify customers at high risk of churning. It provides actionable insights through:
- **Interactive prediction interface** for individual customer scoring
- **Batch prediction** capability for CSV uploads
- **Risk visualization** with probability metrics
- **Professional UI** with modern gradient design and responsive layout

## 📊 Features

### Individual Prediction Mode
- Input customer profile data through an intuitive form
- Get instant churn risk assessment
- View probability scores and risk indicators
- See detailed customer summary

### Batch Prediction Mode
- Upload CSV files with multiple customers
- Process entire datasets at once
- Get churn statistics and risk distribution
- Download prediction results as CSV

### Visual Analytics
- Color-coded risk indicators (High/Low)
- Probability metrics with visual progress bars
- Professional gradient UI with modern design
- Responsive layout for all screen sizes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Customer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
Customer/
├── app.py                          # Main Streamlit application
├── customer_churn_model.pkl        # Trained ML model (pickle file)
├── customer_churn.csv              # Training dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── customer.ipynb                  # Jupyter notebook (optional)
```

## 🔧 Configuration

### Model Features

The model expects the following input features:

**Numerical Features:**
- `age` - Customer age (18-100)
- `income` - Annual income ($0-200,000)
- `credit_score` - Credit score (300-850)
- `transactions_month` - Monthly transactions (0-100)
- `avg_purchase_value` - Average purchase amount ($0-500)
- `days_since_last_login` - Days since last login (0-365)
- `tenure_months` - Customer tenure in months (0-200)
- `num_products` - Number of products owned (1-10)

**Categorical Features:**
- `gender` - Male or Female
- `region` - North, South, East, or West

### Input Data Format (CSV)

For batch predictions, ensure your CSV file contains the above columns:

```csv
age,income,credit_score,transactions_month,avg_purchase_value,days_since_last_login,tenure_months,num_products,gender,region
45,50000,700,20,100,30,24,2,Male,North
52,75000,750,35,150,15,48,3,Female,South
```

## 📈 Model Details

- **Algorithm**: Linear Classification Model (Logistic Regression / SVM)
- **Training Data**: Customer churn dataset
- **Performance**: High accuracy in identifying at-risk customers
- **Output**: Binary classification (Churn/No Churn) with probability scores

## 💡 Usage Examples

### Single Customer Prediction

1. Select "Interactive Form" mode
2. Fill in customer details:
   - Age: 45 years
   - Income: $50,000
   - Credit Score: 700
   - etc.
3. Click "Predict Churn Risk"
4. View results with probability metrics

### Batch Processing

1. Select "Upload CSV" mode
2. Upload a CSV file with customer data
3. Click "Predict Churn for All Customers"
4. Download predictions as CSV file

## 🎨 UI Components

### Hero Section
- Brand title and tagline
- Professional gradient background
- Call-to-action messaging

### Input Forms
- Clean, organized input fields
- Proper input validation
- Helpful labels and tooltips

### Results Display
- Color-coded risk indicators
- Probability metrics in metric boxes
- Progress bar visualization
- Downloadable results

## 🔐 Data Privacy

- No data is stored on the server
- Predictions are computed locally
- CSV uploads are processed in-memory
- No external API calls for predictions

## 📊 Interpretation Guide

### Risk Levels

| Risk Level | Probability | Meaning | Action |
|-----------|------------|---------|--------|
| **HIGH** | 70-100% | Customer likely to churn | Immediate retention needed |
| **MEDIUM** | 40-70% | Uncertain churn risk | Proactive engagement |
| **LOW** | 0-40% | Customer likely to stay | Standard engagement |

## 🛠️ Troubleshooting

### Model File Not Found
**Error:** "Model file not found at: customer_churn_model.pkl"
**Solution:** Ensure the pickle file is in the same directory as app.py

### Feature Name Mismatch
**Error:** "The feature names should match those that were passed during fit"
**Solution:** Check that input CSV has the required column names

### Dependencies Issues
**Error:** "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Run `pip install -r requirements.txt`

## 📝 Requirements

- **streamlit** - Web app framework
- **pandas** - Data manipulation and CSV handling
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning library
- **pickle5** - Serialization support

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Push code to GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app and connect to GitHub repo
4. Select app.py as the main file
5. Deploy with one click

### Deploy Locally with Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```

## 📧 Support

For issues, questions, or improvements:
1. Check the troubleshooting section
2. Verify all requirements are installed
3. Ensure data format matches specifications
4. Check model file integrity

## 📄 License

This project is provided as-is for educational and business use.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Machine Learning powered by [scikit-learn](https://scikit-learn.org)
- Data processing with [Pandas](https://pandas.pydata.org)

## 📈 Version History

- **v1.0** (May 2026) - Initial release with interactive and batch prediction modes

---

**Last Updated:** May 18, 2026
**Status:** Production Ready ✅
