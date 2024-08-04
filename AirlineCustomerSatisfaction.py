import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from PIL import Image
from math import pi
from scipy.stats import zscore

st.set_page_config(
    page_title="Airline Customer Satisfaction Analysis",
    page_icon="✈",
    layout="wide"
)

def get_img_as_base64(image_path):
    """Convert image to base64 format."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return ""

image_path = "background1.jpg"

img = get_img_as_base64(image_path)

# Define the style for background and text color
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/jpeg;base64,{img}");
    background-size: cover; /* Ensures the image covers the entire background */
    background-position: center; /* Centers the image */
    background-repeat: no-repeat; /* Prevents the image from repeating */
    background-attachment: fixed; /* Keeps the image fixed during scrolling */
    min-height: 100vh; /* Ensures the background covers the entire viewport height */
    color: black; /* Sets default text color to black */
}}
[data-testid="stHeader"] {{
    background: rgba(56, 97, 142, 0.3); /* Header background color */
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: rgba(0, 0, 0, 0); /* Transparent background for block wrappers */
    border-radius: 16px;
}}
.st-ds {{
    background-color: rgba(0, 0, 0, 0); /* Transparent background for data sections */
}}
[data-testid="stColorBlock"] {{
    background-color: rgba(0, 0, 0, 0); /* Transparent background for color blocks */
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Airline_customer_satisfaction.csv')
    return df

def preprocess_data(df):
    # Drop features
    df = df.drop(columns=['Arrival Delay in Minutes'])
    
    # Calculate Z-scores for numerical columns
    z_scores = np.abs(zscore(df.select_dtypes(include=['number'])))
    
    # Define the outlier threshold
    threshold = 3
    
    # Filter the DataFrame to remove rows with Z-scores above the threshold
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    for column in df_imputed.select_dtypes(include=['object']).columns:
        df_imputed[column] = label_encoder.fit_transform(df_imputed[column])
    
    return df_imputed

def intro(df_clean):
    st.markdown("<h1 style='text-align: center;'>🛫Airline Customer Satisfaction Analysis🛬</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <h1>Data Science Toolbox</h1>
        <h2>by: 0136275 Debbie Foo Yong Xi , 0136358 Balpreet Kaur</h2>
    </div>
    """, unsafe_allow_html=True)
    video_path = "intro.mp4"
    st.video(video_path)
    st.markdown("<p style='text-align: center; font-size: 14px;'>🎥 Video by Kelly: <a href='https://www.pexels.com/video/people-inside-an-airplane-3740039/'>https://www.pexels.com/video/people-inside-an-airplane-3740039/</a></p>", unsafe_allow_html=True)
    
    st.subheader("📊 Customer Satisfaction Distribution")
    col1, col2 = st.columns([1.5, 2])
    with col1:
        satisfaction_counts = df_clean['satisfaction'].value_counts()
        custom_colors = ['#f0e0a6', '#ff9a9a']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', startangle=70, colors=custom_colors)
        ax.axis('equal')
        st.pyplot(fig)
    with col2:
        st.markdown("""
        <div style='text-align: justify;'>
        
        The pie chart represents the distribution of customer satisfaction levels among airline passengers. 
        
        - **Satisfied:** Represents 55.1% of the passengers. 
        - **Dissatisfied:** Represents 44.9% of the passengers. 

        A larger proportion of satisfied customers shows positive feedback on the overall service, while the portion of dissatisfied customers will later on point to specific issues that need to be addressed.
        
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("📋 Dataset Overview")

    # Display basic information about the dataset
    st.write(f"This dataset contains *{df_clean.shape[0]} rows* and *{df_clean.shape[1]} columns*.")
    st.write(df_clean.head())
   
    # Create a summary DataFrame
    st.write(f"**Dataset Features and Their Data Types:**")

    info_df = pd.DataFrame({
        'Column Name': df_clean.columns,
        'Data Type': [df_clean[col].dtype for col in df_clean.columns],
    })
    
    # Rearranging the columns for better readability
    info_df = info_df[['Column Name', 'Data Type']]

    st.write(info_df)

    with st.expander("**Understanding Dataset Details:**"):
        st.markdown("""
        <div style='text-align: justify;'>

        - **Column Name:** Name of each feature in the dataset.
        - **Data Type:** Type of data (e.g., integer, float, object) for each feature.
        </div>
        """, unsafe_allow_html=True)

def combined_analysis(df_clean):
    # Map numerical values to descriptive labels for the satisfaction column
    df_clean['satisfaction'] = df_clean['satisfaction'].map({0: 'Dissatisfied', 1: 'Satisfied'})

    # Ensure 'satisfaction' is not included in the correlation matrix
    df_numeric = df_clean.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix for numeric data only
    correlation_matrix = df_numeric.corr()

    st.subheader("🌡 Correlation Heatmap")

    # Dropdown
    with st.expander("**What is a Heatmap?**"):
        st.markdown("""
        <div style='text-align: justify;'>
        A heatmap represents how different variables relate to one another using colors. It helps in understanding the strength and direction of relationships between variables.
        
        **Correlation Values:**
        
        **+1: Perfect Positive Correlation –** Both variables move in the same direction. If one variable increases, the other does too.
    
        **-1: Perfect Negative Correlation –** Variables move in opposite directions. If one variable increases, the other decreases.
    
        **0: No Correlation –** There is no relationship between the variables. Changes in one do not affect the other.
        </div>
        """, unsafe_allow_html=True)

    # Generate heatmap
    plt.figure(figsize=(19, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='rainbow', center=0)
    st.pyplot(plt)

    with st.expander("**Understanding the Heatmap**"):
        st.markdown("""
        <div style='text-align: justify;'>
        
        - **Red**: Positive relationships (e.g., when one feature increases, the other does too).
        - **Green**: Negative relationships (e.g., when one feature increases, the other decreases).
        - **Color Intensity**: Stronger colors mean stronger relationships

        **Example of correlations**:
        - **Satisfaction* is most closely linked to **Inflight Entertainment (0.52), Ease of Online Booking (0.43), and Online Support (0.39).** **Ease of Online Booking** is strongly related to **Online Support (0.63)**. Flight Distance and Departure Delay have low correlation with most other features.
    
        </div>
        """, unsafe_allow_html=True)

    # Interactive element
    st.sidebar.subheader("🌍Explore Correlation")
    selected_var = st.sidebar.selectbox(
        "Select a variable to highlight its correlation:",
        df_numeric.columns
    )

    # Display correlation values for the selected variable
    st.markdown(f"### Correlation of {selected_var} with other variables")
    st.markdown("← Please select from **'Explore Correlation'** to edit the correlation and get the table.")
    st.write(correlation_matrix[selected_var])

    # Define pastel colors
    pastel_purple = "#987D9A"
    pastel_purple2 = "#E8C5E5"
    pastel_green = "#BEC6A0"
    pastel_green2 = "#5F6F65"
    pastel_yellow = "#FFD966"
    pastel_yellow2 = "#FFF6BD"
    pastel_blue = "#AAD7D9"
    pastel_blue2 = "#9BB8CD"

    # Get unique values in 'satisfaction' column to check what keys are needed in the palette
    unique_satisfaction_values = df_clean['satisfaction'].unique()

    # Define palette dictionaries based on unique values in 'satisfaction' column
    if set(unique_satisfaction_values) == {'Satisfied', 'Dissatisfied'}:
        satisfaction_palette = {"Satisfied": pastel_blue, "Dissatisfied": pastel_blue2}
        seat_comfort_palette = {"Satisfied": pastel_yellow, "Dissatisfied": pastel_yellow2}
        booking_palette = {"Satisfied": pastel_green, "Dissatisfied": pastel_green2}
        support_palette = {"Satisfied": pastel_purple, "Dissatisfied": pastel_purple2}
    else:
        satisfaction_palette = {0: pastel_blue, 1: pastel_blue2}
        seat_comfort_palette = {0: pastel_yellow, 1: pastel_yellow2}
        booking_palette = {0: pastel_green, 1: pastel_green2}
        support_palette = {0: pastel_purple, 1: pastel_purple2}

    # Feature Importance Analysis
    st.subheader("🔍 Feature Importance Analysis")

    # Explanatory text inside the expander
    with st.expander("**What is Feature Importance Analysis?**"):
        st.markdown("""
        <div style='text-align: justify;'>
        
        **Feature Importance Analysis** is a method used to identify which features or variables in a dataset have the greatest impact on predicting the target variable. By evaluating the significance of each feature, this technique helps us understand which variables are most influential in making predictions. This insight allows us to focus on the most important features, potentially enhancing model performance by selecting and utilizing only the most relevant variables.
        </div>
        """, unsafe_allow_html=True)

    # Prepare features and target
    X = df_clean.drop(columns=['satisfaction'])
    y = df_clean['satisfaction']

    # Ensure the target variable is numeric for the model
    y = y.map({'Dissatisfied': 0, 'Satisfied': 1})

    # Check for missing values
    missing_y = y.isna().sum()
    missing_X = X.isna().sum().sum()

    if missing_y > 0:
        st.error(f"Target variable 'satisfaction' contains {missing_y} missing values.")
    if missing_X > 0:
        st.error(f"Feature data contains {missing_X} missing values.")

    # Handle missing values (example: remove rows with NaNs)
    df_clean = df_clean.dropna(subset=['satisfaction'])
    X = df_clean.drop(columns=['satisfaction'])
    y = df_clean['satisfaction'].map({'Dissatisfied': 0, 'Satisfied': 1})

    # Fit a Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for visualization
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    # Assign a different color to each feature
    colors = sns.color_palette("husl", len(feature_importances))

    # Plot feature importances with different colors for each bar using Plotly
    fig = px.bar(feature_importances.head(10), x='Importance', y='Feature', orientation='h', color='Feature', color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    with st.expander("**Why are we using this?**"):
        st.markdown("""
        <div style='text-align: justify;'>
        Based on the feature importance analysis, we will focus on the top 6 features that contribute most significantly to predicting customer satisfaction. 
        These top features include several numerical attributes, which will be analyzed in more detail. 
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🔍 Top 6 Features Selected from Feature Importance Analysis")
    
    features = [
        "🍿 Inflight Entertainment Ratings",
        "💺 Seat Comfort Ratings",
        "🌐 Ease of Online Booking Ratings",
        "👩🏻‍✈️ On-board Service Ratings",
        "👨🏻‍💻 Online Support Ratings",
        "🍽️ Food and Drink Ratings"
    ]
    
    # User selects a feature
    selected_feature = st.selectbox("Select a feature to explore", features)

    color_maps = {
        'Inflight Entertainment Ratings': {'satisfaction': {0: "#987D9A", 1: "#E8C5E5"}},
        'Seat Comfort Ratings': {'satisfaction': {0: "#BEC6A0", 1: "#5F6F65"}},
        'Ease of Online Booking Ratings': {'satisfaction': {0: "#FFD966", 1: "#FFF6BD"}},
        'On-board Service Ratings': {'satisfaction': {0: "#AAD7D9", 1: "#9BB8CD"}},
        'Online Support Ratings': {'satisfaction': {0: "#BEC6A0", 1: "#5F6F65"}},
        'Food and Drink Ratings': {'satisfaction': {0: "#FFD966", 1: "#FFF6BD"}}
    }
    
    # Plot and description based on selected feature
    if selected_feature == "🍿 Inflight Entertainment Ratings":
        inflight = df_clean[df_clean['Inflight entertainment'] > 0]
        
        col1, col2 = st.columns([2, 1])

        with col1:    
            # Boxplot for Inflight Entertainment Ratings
            fig_box = px.box(inflight, x='satisfaction', y='Inflight entertainment',
                             color='satisfaction',  # Ensure 'color' matches the column name
                             color_discrete_map=color_maps['Inflight Entertainment Ratings']['satisfaction'],
                             labels={'Inflight entertainment': 'Inflight Entertainment Rating'})
            st.plotly_chart(fig_box)
        
        with col2:
            # Display a detailed explanation of the ratings
            st.markdown("""
            <div style='text-align: justify;'>
    
            **Satisfied customers:**
            - Median rating: 4
            - Interquartile range (IQR): 4-5
            - Whiskers: 3-5

            **Dissatisfied customers:**
            - Median rating: 3
            - Interquartile range (IQR): 2-3
            - Whiskers: 1-4
            </div>
            """, unsafe_allow_html=True)

    elif selected_feature == "💺 Seat Comfort Ratings":
        seat_comfort = df_clean[df_clean['Seat comfort'] > 0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Description for Seat Comfort Ratings
            st.markdown("""
            <div style='text-align: justify;'>
            
            **Rating**
            
            1: Satisfied - 9236, Dissatisfied - 11.077k
            
            2: Satisfied - 10.06k, Dissatisfied - 17.797k
            
            3: Satisfied - 10.142k, Dissatisfied - 18.135k
            
            4: Satisfied - 18.139k, Dissatisfied - 9619
            
            5: Satisfied - 17.367k
            
            **Positive Correlation between Seat Comfort and Satisfaction:** Customers who rated seat comfort higher are more likely to be satisfied.
            
            **Negative Correlation between Low Seat Comfort and Satisfaction:** Customers who gave lower seat comfort ratings are more likely to be dissatisfied.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Histogram for Seat Comfort Ratings
            fig_hist = px.histogram(seat_comfort, x='Seat comfort', color='satisfaction',
                                    color_discrete_map=color_maps['Seat Comfort Ratings']['satisfaction'],
                                    category_orders={'Seat comfort': [1, 2, 3, 4, 5]},
                                    labels={'Seat comfort': 'Seat Comfort Rating'})
            st.plotly_chart(fig_hist)
    
    elif selected_feature == "🌐 Ease of Online Booking Ratings":
        online_booking = df_clean[df_clean['Ease of Online booking'] > 0]
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Boxplot for Ease of Online Booking Ratings
            fig_box = px.box(online_booking, x='satisfaction', y='Ease of Online booking',
                             color_discrete_map=color_maps['Ease of Online Booking Ratings']['satisfaction'],
                             labels={'Ease of Online booking': 'Ease of Online Booking Rating'})
            st.plotly_chart(fig_box)
        
        with col2:
            st.markdown("""
            <div style='text-align: justify;'>
            
            **Satisfied Customers:**

            - **Median = 4**, showing  high satisfaction.
            - Short box → ratings are closely clustered around the median.

            **Dissatisfied Customers:**

            - **Median = 3**, indicating lower satisfaction.
            - Taller box → ratings are more spread out (larger IQR).
            - Boxplot extends to a lower rating of 1 → difficulties

            **Interpretation:**

            - **Ease of Booking Affects Satisfaction**: Easier booking is linked to higher satisfaction.
            - **Issues for Dissatisfied Customers**: Many face significant problems with booking.
            </div>
            """, unsafe_allow_html=True)
    
    elif selected_feature == "👩🏻‍✈️ On-board Service Ratings":
        col1, col2 = st.columns([1, 2])
        agg_data = df_clean[df_clean['On-board service'] > 0].groupby(['On-board service', 'satisfaction']).size().reset_index(name='count')
    
        with col1:
            # Description for On-board Service Ratings
            st.markdown("""
            <div style='text-align: justify;'>
    
            **X-axis:** On-board service rating (1 to 5)
            
            **Y-axis:** Number of ratings (0 to 25000)

            **Satisfied customers:** Higher ratings (4, 5) have more responses.
            
            **Dissatisfied customers:** Higher representation in lower ratings (1, 2, 3).
            </div>
            """, unsafe_allow_html=True)

        with col2:            
            # Plot for On-board Service Ratings
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='On-board service', hue='satisfaction', data=df_clean, ax=ax, palette=satisfaction_palette)
            ax.set_xlabel('On-board Service Rating')
            ax.set_ylabel('Count')
            ax.set_xticklabels([1, 2, 3, 4, 5])
            st.pyplot(fig)

    elif selected_feature == "👨🏻‍💻 Online Support Ratings":
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.markdown("""
            <div style='text-align: justify;'>
            
            - **Higher Ratings:** Passengers were generally more satisfied with better online support services.
            - **Lower Ratings:** Indicate that poor online support might negatively impact customer satisfaction.
    
            Enhancing online support services can lead to higher levels of passenger satisfaction.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Online support', hue='satisfaction', data=df_clean, ax=ax, palette=support_palette)
            ax.set_xlabel('Online Support Rating')
            ax.set_ylabel('Count')
            ax.set_xticklabels([1, 2, 3, 4, 5])
            st.pyplot(fig)
            
    elif selected_feature == "🍽️ Food and Drink Ratings":
        col1, col2 = st.columns([1, 2])
    
        with col1:
            # Description for Food and Drink Ratings
            st.markdown("""
            <div style='text-align: justify;'>
    
            **Satisfied Customers:**

            - A majority of customers expressed satisfaction with the food and drink offerings.
            - The highest ratings were concentrated in the upper range of the satisfaction scale.
            
            **Dissatisfied Customers:**
            
            - A smaller but significant portion of customers indicated dissatisfaction.
            - Lower ratings suggest areas where improvements can be made.

            </div>
            """, unsafe_allow_html=True)
    
        with col2:
            food = df_clean[df_clean['Food and drink'] > 0]
            # Plot for Food and Drink Ratings
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Food and drink', hue='satisfaction', data=food, ax=ax, palette=booking_palette)
            ax.set_xlabel('Food and Drink Rating')
            ax.set_xticklabels([1, 2, 3, 4, 5])
            ax.set_ylabel('Count')
            st.pyplot(fig)


def create_individual_radar_charts(data, features, title):
    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    clusters = data['Cluster'].unique()
    num_clusters = len(clusters)

    # Define a list of distinct, sharp colors
    colors = plt.cm.get_cmap("tab10", num_clusters).colors
    
    # Adjust figure size based on number of clusters
    fig_width = 6  
    fig_height = 6  
    
    fig, axs = plt.subplots(1, num_clusters, figsize=(fig_width, fig_height), subplot_kw=dict(polar=True))

    if num_clusters == 1:
        axs = [axs]

    for i, cluster in enumerate(clusters):
        ax = axs[i]
        plt.sca(ax)
        plt.xticks(angles[:-1], features, rotation=45, fontsize=8)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
        plt.ylim(0, 1)

        values = data[data['Cluster'] == cluster][features].mean().values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {int(cluster)}', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_title(f'Cluster {int(cluster)}', size=15, color=colors[i], y=1.1)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)


def clustering_analysis(df_clean):
    st.header("Clustering Analysis for Dissatisfied Customers")
    st.markdown("""
    <div style='text-align: justify;'>
        In this analysis, we will only focus on the dissatisfied group of customers of the dataset to determine the clusters. Numerical features were all standardized so that they contribute equally to the clustering process. Besides, we utilized PCA to address the challenge of high-dimensional data.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    
    # Prepare features and target
    X = df_clean.drop(columns=['satisfaction'])
    y = df_clean['satisfaction']

    features = X.columns.tolist()

    # Standardize all features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Create a DataFrame with scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=features)
    df_scaled['satisfaction'] = df_clean['satisfaction'].values

    # Apply PCA to the entire dataset
    pca = PCA(n_components=2, random_state=111)  # Reduce to 2 dimensions for easy visualization
    principal_components = pca.fit_transform(df_scaled.drop(columns=['satisfaction']))

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['satisfaction'] = df_scaled['satisfaction']
    
    
    # Focus on Dissatisfied Group
    dissatisfied_customers_pca = df_pca[df_pca['satisfaction'] == 0].copy()

    # List to hold SSE values for each number of clusters
    sse = []

    # Loop to calculate SSE for different numbers of clusters
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=111)
        kmeans.fit(dissatisfied_customers_pca[['PC1', 'PC2']])  # Fit on PCA-transformed features
        sse.append(kmeans.inertia_)  # Inertia is the SSE
        
    st.markdown("")
    st.markdown("")
    

    with st.expander("**Elbow Method**"):
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Plot the Elbow Method
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, 11), sse, marker='o', linestyle='--', color='b')
            ax.set_title('Elbow Method for Optimal Number of Clusters')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('SSE')
            ax.set_xticks(range(1, 11))
            ax.grid(True)
        
            st.pyplot(fig)
                
        with col4:
            st.markdown("""
            <div style='text-align: justify;'>
                In this plot, it appears that there is an elbow or “bend” at k = 2 clusters.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("""
            <div style='text-align: justify;'>
                Thus, we will use 2 clusters when fitting our k-means clustering model in the next step.
            </div>
            """, unsafe_allow_html=True)
        
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=111)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PC1', 'PC2']])
    
    # Add the cluster labels back to the original dataframe
    df_clean['Cluster'] = df_pca['Cluster']
    
    # Calculate the percentage of customers in each cluster
    cluster_percentage = (df_pca['Cluster'].value_counts(normalize=True) * 100).reset_index()
    cluster_percentage.columns = ['Cluster', 'Percentage']
    cluster_percentage.sort_values(by='Cluster', inplace=True)
    
    colors = ['#FFD1DC', '#CBAACB']
    
    # Create a horizontal bar plot for cluster distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=colors, ax=ax)
    
    # Adding percentages on the bars
    for index, value in enumerate(cluster_percentage['Percentage']):
        ax.text(value + 0.5, index, f'{value:.2f}%')  

    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    with st.expander("**Cluster Distribution**"):
        column1, column2 = st.columns([1, 2])
        
        with column1:
            st.markdown(f"""
            <div style='text-align: justify;'>
                Cluster 1 had a larger distribution of the overall dataset, consisting of {cluster_percentage[cluster_percentage['Cluster'] == 1]['Percentage'].values[0]:.2f}%. Meanwhile, cluster 0 had a distribution of {cluster_percentage[cluster_percentage['Cluster'] == 0]['Percentage'].values[0]:.2f}% of the dataset.
            </div>
            """, unsafe_allow_html=True)
        
        with column2:
            ax.set_title('Distribution of Customers Across Clusters', fontsize=14)
            ax.set_xticks(ticks=np.arange(0, 50, 5))
            ax.set_xlabel('Percentage (%)')
            st.pyplot(fig)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    with st.expander("**Cluster Characteristics**"):
        # Identify numerical features
        numerical_features = df_clean.select_dtypes(include=['number']).columns.tolist()
    
        # Exclude specific encoded categorical columns
        numerical_features = [f for f in numerical_features if f not in ['CustomerType', 'TypeOfTravel', 'Class', 'satisfaction', 'Cluster']]
    
        # Display cluster characteristics using only numerical data
        cluster_summary = df_clean.groupby('Cluster')[numerical_features].mean()
        st.write(cluster_summary)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader("Radar Charts for Each Cluster")
    # Ensure that the features match the column names in df_clean
    if set(numerical_features).issubset(df_clean.columns):

        # Normalize the features for radar chart
        df_scaled_for_radar = df_clean[df_clean['satisfaction'] == 0].copy()
        scaler_radar = MinMaxScaler()
        df_scaled_for_radar[numerical_features] = scaler_radar.fit_transform(df_scaled_for_radar[numerical_features])

        selected_cluster = st.selectbox("Select a Cluster", options=[0, 1])

        cluster_data = df_scaled_for_radar[df_scaled_for_radar['Cluster'] == selected_cluster]

        # Create side-by-side columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("")
            st.markdown("")
            create_individual_radar_charts(cluster_data, numerical_features, f'Cluster {selected_cluster} Characteristics (Radar Chart)')
        
        with col2:
            if selected_cluster == 0:
                st.markdown("### Overview")
                st.markdown("""
                <div style='text-align: justify;'>
                    Cluster 0 highlighted significant dissatisfaction with ‘Online Support,’ ‘On-board Service,’ and ‘Ease of Online Booking.’ Ratings for ‘Online Boarding,’ ‘Check-in Service,’ and ‘Seat Comfort’ were also low. However, ‘Leg Room’ and ‘Departure/Arrival Time Convenience’ received better ratings, suggesting some aspects of the travel experience met expectations.
                </div>
                """, unsafe_allow_html=True)
                
            elif selected_cluster == 1:
                st.markdown("### Overview")
                st.markdown("""
                <div style='text-align: justify;'>
                    Cluster 1 customers were notably dissatisfied with ‘Check-in Service’ and ‘Gate Location,’ and showed some discontent with ‘Food and Drink’ and ‘Seat Comfort.’ However, they were relatively more satisfied with ‘Ease of Online Booking,’ ‘Online Boarding,’ and ‘Inflight Wi-Fi Service.’
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.write(f"Total Records: {cluster_data.shape[0]}")
            st.write("Standardized Sample Data:")
            st.write(cluster_data.head())

    else:
        st.error("Numerical feature names do not match with DataFrame columns.")
    

    # Randomly sample 10000 observations for metrics computation
    sample_df_pca = df_pca.sample(n=10000, random_state=111)

    # Compute the number of observations
    num_observations = len(sample_df_pca)

    # Separate the features and the cluster labels
    X = sample_df_pca[['PC1', 'PC2']]
    clusters = sample_df_pca['Cluster']

def model_prediction(df_clean):
    st.header("Customer Satisfaction Prediction and Business Insight")

    # Define the mappings for the encoded categorical variables
    type_of_travel_mapping = {0: 'Business Travel 🧳', 1: 'Personal Travel 🏖'}
    customer_type_mapping = {0: 'Disloyal Customer', 1: 'Loyal Customer 🌟'}
    class_mapping = {0: 'Business 🏢', 1: 'Eco+ ✈', 2: 'Eco 🌍'}
    
    # Prepare features and target
    X = df_clean.drop(columns=['satisfaction'])
    y = df_clean['satisfaction']

    # Standardize all features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    
    # Add dropdown and slider for user input
    st.subheader("Select Your Ratings or Categories")
    
    user_input = {}
    
    for feature in X.columns:  
        if feature == 'Age':
            user_input[feature] = st.number_input('Age', min_value=1, max_value=100, value=25)
            st.markdown("- Age Range: Min: 1, Max: 100")
        elif feature == 'Flight Distance':
            user_input[feature] = st.number_input('Flight Distance', min_value=100, max_value=15000, value=100)
            st.markdown("- Flight Distance Range (KM): Min: 100, Max: 15000")
        elif feature == 'Departure Delay in Minutes':
            user_input[feature] = st.number_input('Departure Delay in Minutes', min_value=0, max_value=5000, value=0)
        elif feature == 'TypeOfTravel':
            user_input[feature] = st.radio(feature, options=[0, 1], format_func=lambda x: type_of_travel_mapping[x])
        elif feature == 'CustomerType':
            user_input[feature] = st.radio(feature, options=[0, 1], format_func=lambda x: customer_type_mapping[x])
            st.markdown("- If frequency of visit is more than 3 times, please select Loyal. Else, select disloyal.")
        elif feature == 'Class':
            user_input[feature] = st.radio(feature, options=[0, 1, 2], format_func=lambda x: class_mapping[x])
        else:
            user_input[feature] = st.slider(feature, 1, 5, 3)

    
    # Convert user_input dict to DataFrame
    user_input_df = pd.DataFrame([user_input], columns=X.columns)
    
    # Button for prediction
    if st.button("Predict Satisfaction"):
        # Standardize the user input using the same scaler fitted on X
        user_input_scaled = scaler.transform(user_input_df)
        
        # Train RandomForest with all features
        rf = RandomForestClassifier(random_state=111)
        rf.fit(scaled_features, y)

        # Load images
        satisfied_image = Image.open("satisfied.png")
        dissatisfied_image = Image.open("dissatisfied.png")
        smooth_check_in = Image.open("human-touch.jpg")
        express_travel = Image.open("express_travel.png")

        # Predict satisfaction using the trained RandomForest
        satisfaction_prediction = rf.predict(user_input_scaled)[0]

        # Display the corresponding image based on the prediction
        if satisfaction_prediction == 1:
            st.image(satisfied_image, caption="Satisfied")
            st.write("")
            st.write("*We're delighted to hear that you enjoyed your experience with us! Thank you for choosing our airline, and we can't wait to welcome you back on your next journey!*")

        else:
            st.image(dissatisfied_image, caption="Dissatisfied")

        if satisfaction_prediction == 0:
            # Fit KMeans on all features
            kmeans = KMeans(n_clusters=2, random_state=111)
            kmeans.fit(scaled_features)
            cluster_prediction = kmeans.predict(user_input_scaled)
            st.write(f"**Predicted Cluster: {cluster_prediction[0]}**")
            
            with st.expander("**Recommendations Based on Cluster**"):
                cluster = cluster_prediction[0]
                # Recommendations for different clusters
                if cluster == 0:
                    st.markdown("To address the specific needs of Cluster 0, we propose the *Express Travel Package*.")
                    st.markdown("")
                    st.image(express_travel, caption="-")
                    st.markdown("This package includes:")
                    st.markdown("**1. Enhanced Online Support and Booking:**")
                    st.markdown("   - Upgrade online support with 24/7 live chat and comprehensive FAQs.")
                    st.markdown("   - Simplify the booking process for a smoother user experience.")
                    st.markdown("**2. Improved On-Board Experience:**")
                    st.markdown("   - Offer personalized in-flight service with better meal options.")
                    st.markdown("   - Enhance seating comfort with ergonomically designed seats and extra legroom.")
                    st.markdown("**3. Convenience Enhancements:**")
                    st.markdown("   - Provide priority boarding and faster baggage handling to reduce waiting times.")
                        
                elif cluster == 1:
                    st.markdown("To meet the needs of Cluster 1, we propose the *Smooth Check-in Package*.")
                    st.markdown("")
                    st.image(smooth_check_in, caption="-", width=450)
                    st.markdown("This package includes:")
                    st.markdown("**1. Efficient Check-In and Gate Convenience:**")
                    st.markdown("   - Enhance check-in with more self-service kiosks and dedicated counters.")
                    st.markdown("   - Ensure convenient gate locations with real-time updates and notifications.")
                    st.markdown("**2. Enhanced On-Board Amenities:**")
                    st.markdown("   - Improve food and beverage options with diverse dietary choices.")
                    st.markdown("   - Upgrade seat comfort with better lumbar support and adjustable features.")
                    st.markdown("**3. Additional Comfort and Convenience:**")
                    st.markdown("   - Provide complimentary inflight Wi-Fi and priority boarding.")
                
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
    
    # Train model
    rf = RandomForestClassifier(random_state=111)
    rf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = rf.predict(X_test)
    
page_names_to_funcs = {
    "Introduction": intro,
    "Flight Experience Analysis": combined_analysis,
    "Clustering Analysis (Dissatisfied Group)": clustering_analysis,
    "Model Prediction": model_prediction
}

df = load_data()
df_clean = preprocess_data(df)

# Ensure the function is called with the right arguments
demo_name = st.sidebar.selectbox("**📄Choose a page**", page_names_to_funcs.keys())
page_names_to_funcs[demo_name](df_clean)