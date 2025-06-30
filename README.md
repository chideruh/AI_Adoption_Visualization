This project is a comprehensive data analytics and machine learning project analyzing global AI tool adoption patterns across industries and regions (data collected from 2023- early 2025). It provides predicitve inisghts into AI adoption trends using advanced data visualization and statistical modelling all done in Python.

# **Project Overview**
This project analyzes the [Global AI Tool Adoption Across Industries and Regions]((https://www.kaggle.com/datasets/tfisthis/global-ai-tool-adoption-across-industries)) dataset from Kaggle, providing insihgts into: 
- AI tool adoption rates by industry and region
- Usage partterns across different demographics
- Predicitive modeling ofr future adoption trends
- Static visualiation showcasing information in various aesthetic formats
- Statistical analysis fo user feedback sentiment

# **Dataset Information**
Source: [Github](https://www.kaggle.com/datasets/tfisthis/global-ai-tool-adoption-across-industries)
| Column  | Description | Type |
|----------|----------|----------|
| country    | Country location     | String|
| industry    | Industry sector     | String|
| ai_tool    | AI tool name (ChatGPT, Bard, etc)     | String|
| adoption_rate    | Adoption percentage     | Float|
| daily_active_users    | Estimate Daily Active Users     | Integer|
| year    | Data recording year (2023, 2024, 2025)     | Integer|
| user_feedback    | User experience feedback (up to 150 characters)     |String|
| age_group    | User age categories     | String|
| company_size    | Organization size( Startup, SME, Enterprose)     | String|

# **Project Specs**
- Python 3.8+
- pip or conda package manager

### **Libraries and APIs Used**
- Pandas: Data manipulation and analysis
- Numpy: Numerical computing
- Scikit-learn: Machine learning
- Seaborn: Statistical data visualization
- Matplotlib: Static plotting
- Scipy: Scientific computing
- Statsmodels: Statistical modeling

# **Key Analysis Components**
## 1. **Exploratory Data Analysis(EDA)**
   === STATISTICAL SUMMARY ===
count == 145000.000000

mean == 49.873025

std == 28.842523

min == 0.000000

25% == 24.930000

50%  == 49.760000

75% == 74.840000

max == 100.000000

Name: adoption_rate, dtype: float64

## 2. **Correlation Analysis**
Correlation between adoption_rate and daily_active_users:

Correlation coefficient: 0.0022

P-value: 0.4102

Statistically significant: No

Chi-square test (Industry vs Adoption Category):

Chi-square statistic: 15.8907

P-value: 0.7758

Degrees of freedom: 21

## 3. **Predictive Modeling**
   Model Performance Comparison:
| Model | R2 Score | RMSE |
|----------|----------|----------|
| Linear Regression   | 0.000021     | 28.826764|
| Random Forest   | -0.080025     | 29.958319|

## 4. **Customer Segmentation Analysis**
   Cluster Summary:
|cluster      | adoption_rate|  daily_active_users|  engagement_score|       
|-------------|--------------|--------------------|------------------|
|0            |24.017372|         2448.744648|        542.474395|
|1            |73.359978|        2711.248739 |      1940.632971|
|2            |76.208598|         7654.859893|       5774.051172|
|3            |26.152559|        7353.653767 |      1879.072619|

# **Business Intelligence Report**
 ### **Key Business Insights**
- Total Market Size: **730,698,889 daily active users**
- Average Adoption Rate: **49.9%**
- Top Performing Country: **Germany**
- Top Performing Industry: **Agriculture**
- Market Leader (Tool): **ChatGPT**
- Annual Growth Rate:** -0.4%**

### **Strategic Recommendations**
- Focus expansion efforts on Germany market
- Prioritize Agriculture industry vertical
- Investigate low-adoption segments for growth opportunities
