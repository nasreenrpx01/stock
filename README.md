### README for Stock Movement Prediction Model Using Reddit Data  

---

#### **Overview**  
This project aims to predict stock movement based on sentiment and engagement data from Reddit's "stocks" subreddit. The pipeline includes scraping Reddit posts, preprocessing data, sentiment analysis, model training using XGBoost, and model evaluation.  

The final model achieved:  
- **Mean Absolute Error (MAE)**: 36.18  
- **Root Mean Squared Error (RMSE)**: 47.96  
- **R² Score**: 94.47%  

---

#### **Features**  

1. **Data Scraping**: 
   - Utilized the `PRAW` library to collect data (titles, scores, comments, timestamps) from the "stocks" subreddit.  
2. **Preprocessing**: 
   - Extracted temporal features (day of the week, time of day, weekend indicator).  
   - Calculated sentiment scores using the `VADER Sentiment Analysis` tool.  
   - Created additional features, such as the comment-to-score ratio and weighted sentiment.  
3. **Exploratory Data Analysis (EDA)**: 
   - Visualized distributions, correlations, and feature relationships.  
4. **Model Training**: 
   - Used XGBoost for regression.  
   - Optimized hyperparameters through GridSearchCV.  
5. **Evaluation**:  
   - Assessed model accuracy using MAE, RMSE, and R² Score.  

---

#### **Setup Instructions**  

1. **Prerequisites**:  
   - Python (>=3.7)  
   - Required libraries:  
     ```
     praw  
     pandas  
     numpy  
     scikit-learn  
     xgboost  
     vaderSentiment  
     matplotlib  
     seaborn  
     ```

   - Install dependencies using:  
     ```bash
     pip install praw pandas numpy scikit-learn xgboost vaderSentiment matplotlib seaborn
     ```  

2. **Reddit API Configuration**:  
   - Register for API credentials on the [Reddit Developer Portal](https://www.reddit.com/prefs/apps).  
   - Add your `client_id`, `client_secret`, and `user_agent` to the script:  
     ```python
     reddit = praw.Reddit(
         client_id="YOUR_CLIENT_ID",
         client_secret="YOUR_CLIENT_SECRET",
         user_agent="YOUR_USER_AGENT"
     )
     ```

3. **Run the Script**:  
   - Ensure all dependencies are installed and API credentials are configured.  
   - Execute the Jupyter Notebook or Python script:  
     ```bash
     jupyter notebook stock_movement_prediction.ipynb
     ```  
     or  
     ```bash
     python stock_movement_prediction.py
     ```  

---

#### **Pipeline Overview**  

1. **Data Scraping**:  
   - Scraped 1,000 posts from Reddit, extracting features like `title`, `score`, `num_comments`, and `created`.  

2. **Feature Engineering**:  
   - Temporal: Extracted `day_of_week`, `is_weekend`, and `time_of_day`.  
   - Sentiment: Used VADER to compute `sentiment_score` (compound), `positive_sentiment`, `negative_sentiment`, and `neutral_sentiment`.  
   - Ratios: Computed `comment_to_score_ratio` and `weighted_sentiment`.  

3. **Modeling and Evaluation**:  
   - Split data into training (80%) and test (20%) sets.  
   - Standardized features using `StandardScaler`.  
   - Trained an XGBoost regression model with hyperparameter tuning (learning rate, max depth, estimators, etc.).  

---

#### **Results**  

| Metric       | Value        |  
|--------------|--------------|  
| **MAE**      | 36.18        |  
| **RMSE**     | 47.96        |  
| **R² Score** | 94.47%       |  

The model demonstrates good performance without significant signs of overfitting.  

---

#### **Files**  

- `stock_movement_prediction.ipynb`: Jupyter Notebook containing the full pipeline.  
- `README.md`: This documentation.  
- `requirements.txt`: List of all dependencies.  

---

#### **Future Work**  

- **Enhance Sentiment Analysis**: Experiment with advanced NLP models like BERT.  
- **Integrate Financial Data**: Combine Reddit data with stock price trends for a more robust prediction model.  
- **Optimize Model**: Test other algorithms (LightGBM, CatBoost) and refine hyperparameters further.  

---

#### **Author**  
Nasreen Fatima  
2024
