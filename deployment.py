import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from streamlit_option_menu import option_menu
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from io import BytesIO

os.chdir(r"C:\Users\ASUS\Desktop\clg\Projects")
st.set_page_config(
    page_title="Premier League Prediction",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)
st.markdown(f'''   <style>
       .stApp {{
       background-image: url('https://i.pinimg.com/originals/88/b0/4e/88b04e07ca32555a2876f22d144bb927.jpg');
       background-attachment: fixed;
        background-size: cover
       }}
       </style> 
        
       ''',   unsafe_allow_html=True)
# Load training data and prediction data
training_data = pd.read_csv('training.csv')
prediction_data = pd.read_csv('predictionn_2023.csv')
eda_data=pd.read_excel('PL Table Prediction PDS.xlsx')

# Function to train the model
def train_model(train_data):
    features = ['team_rating', 'opp_rating', 'att_rating', 'def_rating', 'mid_rating',
                'Opponent Code', 'Team Code', 'Venue Code', 'Hour']
    X = train_data[features]
    y = train_data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=110, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Function to load the saved model and apply predictions
def predict_results(prediction_data, model):
    features = ['team_rating', 'opp_rating', 'att_rating', 'def_rating', 'mid_rating',
                'Opponent Code', 'Team Code', 'Venue Code', 'Hour']
    prediction_data['Predicted_Result'] = model.predict(prediction_data[features])
    return prediction_data

# Function to create the points table
def create_points_table(prediction, all_teams):
    points_table = {team: 0 for team in all_teams}

    for index, row in prediction.iterrows():
        if row['Predicted_Result'] == 1:
            points_table[row['Team']] += 3
        elif row['Predicted_Result'] == 0:
            points_table[row['Team']] += 1
            points_table[row['Opponent']] += 1

    points_df = pd.DataFrame(points_table.items(), columns=['Team', 'Points'])
    points_df['Wins'] = points_df['Points'] // 3
    points_df['Draws'] = (points_df['Points'] % 3) // 1
    points_df['Losses'] = 38 - (points_df['Wins'] + points_df['Draws'])
    points_df['Total Matches'] = points_df['Wins'] + points_df['Draws'] + points_df['Losses']
    points_df = points_df.sort_values(by='Points', ascending=False).reset_index(drop=True)
    points_df.index = points_df.index + 1

    return points_df

def load_and_filter_data(file_path, selected_position):
    df = pd.read_excel(file_path)
    if selected_position:
        filtered_data = df[df['pos'] == selected_position]
    else:
        filtered_data = df
    return filtered_data
# Streamlit app
# Create sidebar for team selection

xls = pd.ExcelFile(r"PL Master Data.xlsx")
valid_teams = xls.sheet_names
with st.sidebar:        
    app = option_menu (
        menu_title='Options',
        options=['Predicting 23-24','Team Data','Previous 2 Seasons Stats'],
    )

# Filter out invalid team names and display the selector


# Predict and display points table
model = train_model(training_data)
if app== "Predicting 23-24":
    st.title("EPL 23-24 Season Points Table Predictor")
    st.markdown("### Excitement is in the air as we predict the EPL 23-24 season winner. Which team will claim the coveted title?")
    st.divider()

    if st.button("Predict"):
        with st.empty():
            for i in range(5):
                st.write("Predicting 2023 table...")
                time.sleep(1)

            st.title("Points Table 2023")
            all_teams = prediction_data['Team'].unique()  # Replace with the actual column name for team
            prediction_with_results = predict_results(prediction_data, model)
            points_table = create_points_table(prediction_with_results, all_teams)
            st.write(points_table)

# Display team players data in the main content area
if app== "Team Data":
    st.title("Team Player's Information")
    selected_team = st.selectbox('Select Team', valid_teams)
    st.write("Team:", selected_team)
    team_data = pd.read_excel(r"PL Master Data(1).xlsx", sheet_name=selected_team)
    st.write(team_data)

if app == "Previous 2 Seasons Stats":
    st.subheader("Goal scored by each team in the last 2 years")

    # Generate sample data for visualization (replace this with your actual data)
    team_goals = training_data.groupby('Team')['GF'].sum().reset_index()

    # Sort the data by total goals scored in descending order
    team_goals = team_goals.sort_values(by='GF', ascending=False)

    # Create a bar plot to show how much each team has scored
    fig = px.bar(team_goals, x='GF', y='Team', orientation='h', color='GF', text='GF',
                 color_continuous_scale='sunset')

    # Show the plot
    st.plotly_chart(fig)

    st.subheader('Venue vs. Results')
    # Generate sample data for visualization (replace this with your actual data)
    venue_results = training_data.groupby(['Venue', 'Result']).size().reset_index(name='Count')

    # Create a bar plot to show venue vs. results for each team
    fig = px.bar(venue_results, x='Venue', y='Count', color='Result', barmode='group',
                 color_discrete_map={'W': '#00A3E0', 'L': '#EF3E42'})

    fig.update_layout(
        xaxis_title='Venue',
        yaxis_title='Count'
    )
    st.plotly_chart(fig)

    st.subheader("Goals For (GF) and Goals Against (GA) Comparison")
    trace1 = go.Bar(
        y=training_data['Team'],
        x=training_data['GF'],
        name='Goals For (GF)',
        orientation='h',
        marker=dict(color='#00A3E0')
    )

    trace2 = go.Bar(
        y=training_data['Team'],
        x=training_data['GA'],
        name='Goals Against (GA)',
        orientation='h',
        marker=dict(color='#EF3E42')
    )

    # Create layout
    layout = go.Layout(
        barmode='group',
        xaxis=dict(title='Goals'),
        yaxis=dict(title='Team')
    )

    # Create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Show the plot
    st.plotly_chart(fig)

    # Create trace
    st.header("Possession Team Wise per 90 minutes")
    trace = go.Bar(
        x=eda_data['Team'],
        y=eda_data['Passes'],
        marker=dict(color='#00A3E0')
    )

    # Create layout
    layout = go.Layout(
        xaxis=dict(title='Team', tickangle=45),
        yaxis=dict(title='Passes per 90')
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot
    st.plotly_chart(fig)

    # Count occurrences of each formation
    st.header("Formation vs. Result")
    formation_counts = training_data['Formation'].value_counts()

    # Create trace
    trace = go.Bar(
        x=formation_counts.index,
        y=formation_counts.values,
        marker=dict(color='#EF3E42')
    )

    # Create layout
    layout = go.Layout(
        xaxis=dict(title='Formation', tickangle=45),
        yaxis=dict(title='Count')
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot
    st.plotly_chart(fig)
    st.header("Teamwise Goals vs. Assists")
    fig = px.scatter(eda_data, x='Goals', y='Assists', color='Team',
                 labels={'Goals': 'Goals', 'Assists': 'Assists'}, color_discrete_sequence=px.colors.qualitative.Set1)

    # Adding text labels
    for i in range(len(eda_data)):
        fig.add_annotation(x=eda_data['Goals'][i] + 0.1, y=eda_data['Assists'][i], text=eda_data['Team'][i], showarrow=False, font=dict(size=8))

    # Update the layout
    fig.update_layout(xaxis_title='Goals', yaxis_title='Assists', legend_title='Team')

    # Display the plot using Streamlit
    st.plotly_chart(fig)
    
    # Create a bar plot using Plotly for Formation vs Result
    plt.figure(figsize=(16, 8))
    st.title('Formation vs Result')
    sns.set_theme(style="whitegrid")
    sns.countplot(x='Formation', hue='Result', data=training_data, palette='Set2')
    plt.xlabel('Formation', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.gca().set_facecolor('black')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Display the plot in Streamlit
    st.image(buffer, caption='Formation vs Result', use_column_width=True)
    
    team_ratings = training_data.groupby('Team')[['team_rating']].mean().reset_index()

# Sorting the values to get a clearer plot
    team_ratings = team_ratings.sort_values(by='team_rating', ascending=False)
    st.header("Team Ratings")
    # Plotting the bar plot using Plotly
    fig = px.bar(team_ratings, x='team_rating', y='Team', orientation='h', text='team_rating', color='team_rating')
    fig.update_layout(xaxis_title='Team Rating', yaxis_title='Team', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    fig.update_traces(marker_color='cyan', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)

    # Displaying the plot using Streamlit
    st.plotly_chart(fig)
    
    # Calculate the average values for team_att, team_def, and team_mid for each team
    average_ratings = training_data.groupby('Team')[['team_att', 'team_def', 'team_mid']].mean().reset_index()
    st.header("Comparison of Ratings for Each Team")

    # Reshape the data for visualization
    melted_ratings = average_ratings.melt(id_vars='Team', var_name='Rating Type', value_name='Rating Value')

    # Create an interactive bar plot using Plotly
    fig = px.bar(melted_ratings, x='Team', y='Rating Value', color='Rating Type', barmode='group')

    # Use the st.plotly_chart function to display the figure in Streamlit
    st.plotly_chart(fig)


      