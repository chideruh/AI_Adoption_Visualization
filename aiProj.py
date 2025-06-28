import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AIAdoptionAnalyzer:
    """
    Comprehensive AI Tool Adoption Analysis Class
    
    This class performs advanced statistical analysis on AI tool adoption data,
    including predictive modeling, clustering, and business intelligence insights.
    """
    
    def __init__(self, csv_file_path):
        """Initialize the analyzer with data loading and basic preprocessing."""
        self.df = pd.read_csv(csv_file_path)
        self.clean_data()
        self.create_features()
        
    def clean_data(self):
        """Perform data cleaning and quality assessment."""
        print("=== DATA QUALITY ASSESSMENT ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values per column:\n{self.df.isnull().sum()}")
        print(f"Data types:\n{self.df.dtypes}")
        
        # Handle missing values
        numeric_columns = ['adoption_rate', 'daily_active_users']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Clean text columns
        text_columns = ['country', 'industry', 'ai_tool', 'user_feedback', 'age_group', 'company_size']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[col] = self.df[col].str.strip()
        
        # Remove outliers using IQR method
        Q1 = self.df['adoption_rate'].quantile(0.25)
        Q3 = self.df['adoption_rate'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(self.df)
        self.df = self.df[(self.df['adoption_rate'] >= lower_bound) & 
                         (self.df['adoption_rate'] <= upper_bound)]
        outliers_removed = outliers_before - len(self.df)
        print(f"Outliers removed: {outliers_removed}")
        
    def create_features(self):
        """Feature engineering for advanced analysis."""
        # Create adoption categories
        self.df['adoption_category'] = pd.cut(self.df['adoption_rate'], 
                                            bins=[0, 25, 50, 75, 100], 
                                            labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Create engagement metrics
        if 'daily_active_users' in self.df.columns and 'adoption_rate' in self.df.columns:
            self.df['engagement_score'] = (self.df['daily_active_users'] * 
                                         self.df['adoption_rate'] / 100)
        
        # Time-based features
        if 'year' in self.df.columns:
            self.df['years_since_2023'] = self.df['year'] - 2023
        
        print("Feature engineering completed.")
        
    def perform_eda(self):
        """Comprehensive Exploratory Data Analysis."""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI Tool Adoption - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        #Distribution of adoption rates
        axes[0,0].hist(self.df['adoption_rate'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Adoption Rates')
        axes[0,0].set_xlabel('Adoption Rate (%)')
        axes[0,0].set_ylabel('Frequency')
        
        #Top countries by adoption
        country_adoption = self.df.groupby('country')['adoption_rate'].mean().sort_values(ascending=False).head(10)
        country_adoption.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Top 10 Countries by Average Adoption Rate')
        axes[0,1].set_ylabel('Average Adoption Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        #Industry comparison
        industry_stats = self.df.groupby('industry')['adoption_rate'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        industry_stats['mean'].plot(kind='bar', ax=axes[0,2], color='lightgreen', yerr=industry_stats['std'])
        axes[0,2].set_title('Industry Adoption Rates (with std dev)')
        axes[0,2].set_ylabel('Average Adoption Rate (%)')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        #AI Tool popularity
        tool_usage = self.df.groupby('ai_tool')['daily_active_users'].sum().sort_values(ascending=False)
        tool_usage.plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Total Daily Active Users by AI Tool')
        axes[1,0].set_ylabel('Daily Active Users')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        #Company size impact
        if 'company_size' in self.df.columns:
            size_adoption = self.df.groupby('company_size')['adoption_rate'].mean()
            axes[1,1].pie(size_adoption.values, labels=size_adoption.index, autopct='%1.1f%%')
            axes[1,1].set_title('Adoption Rate by Company Size')
        
        #Age group analysis
        if 'age_group' in self.df.columns:
            age_adoption = self.df.groupby('age_group')['adoption_rate'].mean().sort_values(ascending=False)
            age_adoption.plot(kind='bar', ax=axes[1,2], color='purple', alpha=0.7)
            axes[1,2].set_title('Adoption Rate by Age Group')
            axes[1,2].set_ylabel('Average Adoption Rate (%)')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("\n=== STATISTICAL SUMMARY ===")
        print(self.df['adoption_rate'].describe())
        
    def correlation_analysis(self):
        """Advanced correlation and statistical testing."""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Create correlation matrix for numeric variables
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix - AI Adoption Metrics')
        plt.show()
        
        # Statistical significance testing
        if 'daily_active_users' in self.df.columns:
            corr_coef, p_value = pearsonr(self.df['adoption_rate'], self.df['daily_active_users'])
            print(f"Correlation between adoption_rate and daily_active_users:")
            print(f"Correlation coefficient: {corr_coef:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Chi-square test for categorical variables
        if 'industry' in self.df.columns:
            contingency_table = pd.crosstab(self.df['industry'], self.df['adoption_category'])
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-square test (Industry vs Adoption Category):")
            print(f"Chi-square statistic: {chi2:.4f}")
            print(f"P-value: {p_val:.4f}")
            print(f"Degrees of freedom: {dof}")
        
    def predictive_modeling(self):
        """Advanced predictive modeling with multiple algorithms."""
        print("\n=== PREDICTIVE MODELING ===")
        
        # Prepare features for modeling
        le = LabelEncoder()
        features_df = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['country', 'industry', 'ai_tool', 'age_group', 'company_size']
        for col in categorical_cols:
            if col in features_df.columns:
                features_df[f'{col}_encoded'] = le.fit_transform(features_df[col])
        
        # Select features for prediction
        feature_cols = [col for col in features_df.columns if col.endswith('_encoded')] + \
                      ['daily_active_users', 'year']
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols]
        y = features_df['adoption_rate']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model 1: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        # Model 2: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        # Results comparison
        results_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'R² Score': [lr_r2, rf_r2],
            'RMSE': [lr_rmse, rf_rmse]
        })
        
        print("Model Performance Comparison:")
        print(results_df.to_string(index=False))
        
        # Feature importance (Random Forest)
        if len(feature_cols) > 0:
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.xlabel('Importance Score')
            plt.show()
        
        # Save models for future use
        self.best_model = rf_model if rf_r2 > lr_r2 else lr_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        
        return results_df
    
    def customer_segmentation(self):
        """Perform customer segmentation using clustering."""
        print("\n=== CUSTOMER SEGMENTATION ANALYSIS ===")
        
        # Prepare data for clustering
        cluster_features = ['adoption_rate', 'daily_active_users']
        if 'engagement_score' in self.df.columns:
            cluster_features.append('engagement_score')
        
        cluster_data = self.df[cluster_features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()
        
        # Perform clustering with optimal k (let's use 4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to dataframe
        cluster_df = cluster_data.copy()
        cluster_df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_summary = cluster_df.groupby('cluster')[cluster_features].mean()
        print("Cluster Summary:")
        print(cluster_summary)
        
        # Visualize clusters
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(cluster_df['adoption_rate'], cluster_df['daily_active_users'], 
                            c=cluster_df['cluster'], cmap='viridis')
        plt.xlabel('Adoption Rate (%)')
        plt.ylabel('Daily Active Users')
        plt.title('Customer Segments')
        plt.colorbar(scatter)
        
        plt.subplot(1, 2, 2)
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        plt.pie(cluster_counts.values, labels=[f'Segment {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%')
        plt.title('Segment Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return cluster_summary
    
    def business_intelligence_report(self):
        """Generate comprehensive business intelligence insights."""
        print("\n=== BUSINESS INTELLIGENCE REPORT ===")
        
        # Market opportunity analysis
        total_market = self.df['daily_active_users'].sum()
        avg_adoption = self.df['adoption_rate'].mean()
        
        # Top performing segments
        country_performance = self.df.groupby('country').agg({
            'adoption_rate': 'mean',
            'daily_active_users': 'sum'
        }).sort_values('adoption_rate', ascending=False)
        
        industry_performance = self.df.groupby('industry').agg({
            'adoption_rate': 'mean',
            'daily_active_users': 'sum'
        }).sort_values('adoption_rate', ascending=False)
        
        # Growth trends
        if 'year' in self.df.columns:
            yearly_growth = self.df.groupby('year')['adoption_rate'].mean()
            growth_rate = ((yearly_growth.iloc[-1] - yearly_growth.iloc[0]) / yearly_growth.iloc[0]) * 100
        
        # Generate insights
        insights = {
            'Total Market Size': f"{total_market:,.0f} daily active users",
            'Average Adoption Rate': f"{avg_adoption:.1f}%",
            'Top Performing Country': country_performance.index[0],
            'Top Performing Industry': industry_performance.index[0],
            'Market Leader (Tool)': self.df.groupby('ai_tool')['daily_active_users'].sum().idxmax()
        }
        
        if 'year' in self.df.columns:
            insights['Annual Growth Rate'] = f"{growth_rate:.1f}%"
        
        print("KEY BUSINESS INSIGHTS:")
        for key, value in insights.items():
            print(f"• {key}: {value}")
        
        # Recommendations
        print("\nSTRATEGIC RECOMMENDATIONS:")
        print("• Focus expansion efforts on", country_performance.index[0], "market")
        print("• Prioritize", industry_performance.index[0], "industry vertical")
        print("• Investigate low-adoption segments for growth opportunities")
        
        return insights

def main():
    """Main execution function - replace 'your_data.csv' with your actual file path."""
    
    # Initialize analyzer (you'll need to update this path)
    print("AI Tool Adoption Analysis - Professional Portfolio Project")
    print("=" * 60)
    
    # Note: Replace with your actual CSV file path
    file_path = "/Users/chideranwana/Downloads/ai_adoption_dataset.csv"  # Update this path
    
    try:
        # Initialize analyzer
        analyzer = AIAdoptionAnalyzer(file_path)
        
        # Perform comprehensive analysis
        analyzer.perform_eda()
        analyzer.correlation_analysis()
        model_results = analyzer.predictive_modeling()
        segmentation_results = analyzer.customer_segmentation()
        business_insights = analyzer.business_intelligence_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - Portfolio Ready!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please update the file_path variable with your actual CSV file location.")
        print("\nExample usage:")
        print("1. Save your CSV file in the same directory as this script")
        print("2. Update the file_path variable to match your filename")
        print("3. Run the script again")

if __name__ == "__main__":
    main()

# Additional utility functions for advanced analysis

def perform_time_series_analysis(df, date_column, value_column):
    """
    Perform time series analysis for forecasting.
    Useful for predicting future adoption trends.
    """
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()
    
    # Decompose time series
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df[value_column], model='additive', period=12)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residuals')
    plt.tight_layout()
    plt.show()

def calculate_business_metrics(df):
    """
    Calculate key business metrics for executive reporting.
    """
    metrics = {}
    
    # Market penetration
    if 'adoption_rate' in df.columns:
        metrics['market_penetration'] = df['adoption_rate'].mean()
    
    # Customer acquisition cost (mock calculation)
    if 'daily_active_users' in df.columns:
        metrics['avg_daily_users'] = df['daily_active_users'].mean()
    
    # Growth rate
    if 'year' in df.columns:
        yearly_adoption = df.groupby('year')['adoption_rate'].mean()
        if len(yearly_adoption) > 1:
            growth_rate = (yearly_adoption.iloc[-1] / yearly_adoption.iloc[0] - 1) * 100
            metrics['growth_rate'] = growth_rate
    
    return metrics

def export_results_to_excel(analyzer, filename="ai_adoption_analysis_results.xlsx"):
    """
    Export analysis results to Excel for stakeholder sharing.
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Raw data
        analyzer.df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        # Summary statistics
        summary_stats = analyzer.df.describe()
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Country performance
        country_perf = analyzer.df.groupby('country')['adoption_rate'].agg(['mean', 'std', 'count'])
        country_perf.to_excel(writer, sheet_name='Country_Performance')
        
        # Industry analysis
        industry_perf = analyzer.df.groupby('industry')['adoption_rate'].agg(['mean', 'std', 'count'])
        industry_perf.to_excel(writer, sheet_name='Industry_Analysis')
    
    print(f"Results exported to {filename}")

