#Ran on google collab

print("🚀 Starting Iris ML Classifier...")
print("📦 Installing packages...")

!pip install streamlit -q
!pip install pandas numpy scikit-learn plotly matplotlib seaborn -q

print("✅ Packages installed!")
print("📝 Creating app...")

# Create app file
app_content = '''import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")
st.title("🌸 Iris Flower Classification ML Project")
st.markdown("**Predict iris species using machine learning!**")

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data", "Visualization", "Predict"])

if page == "Data":
    st.header("📊 Iris Dataset")
    st.write(f"**Samples:** {len(df)} | **Features:** {len(df.columns)-1} | **Classes:** {df['species'].nunique()}")
    st.dataframe(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistics")
        st.write(df.describe())
    with col2:
        st.subheader("Class Distribution")
        st.bar_chart(df['species'].value_counts())

elif page == "Visualization":
    st.header("📈 Data Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", df.columns[:-1])
    with col2:
        y_axis = st.selectbox("Y-axis", df.columns[:-1])
    
    fig = px.scatter(df, x=x_axis, y=y_axis, color='species',
                     title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("🔮 Make Prediction")
    
    # Train model
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    st.success(f"**Model Accuracy:** {accuracy:.1%}")
    
    # Input sliders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    with col2:
        sw = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
    with col3:
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    with col4:
        pw = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)
    
    if st.button("🌺 Predict Species", type="primary"):
        prediction = model.predict([[sl, sw, pl, pw]])[0]
        st.balloons()
        st.success(f"**Predicted Species:** {prediction}")
        
        # Show probabilities
        probs = model.predict_proba([[sl, sw, pl, pw]])[0]
        prob_df = pd.DataFrame({
            'Species': iris.target_names,
            'Probability': probs
        })
        st.subheader("Prediction Confidence")
        st.bar_chart(prob_df.set_index('Species'))

st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn | Iris Dataset")
'''

with open('iris_app.py', 'w') as f:
    f.write(app_content)

print("✅ App created!")
print("🚀 Getting public URL...")

# Get the public URL using Colab's method
try:
    from google.colab.output import eval_js
    import IPython
    
    # Get public URL
    public_url = eval_js("google.colab.kernel.proxyPort(8501)")
    
    print("\n" + "="*70)
    print("🎉 YOUR APP IS READY!")
    print("="*70)
    print(f"\n🌐 **CLICK THIS LINK:**")
    print(f"👉 {public_url}")
    print("\n" + "="*70)
    
    # Display as clickable link
    display(IPython.display.HTML(f'<h3><a href="{public_url}" target="_blank">🚀 CLICK HERE TO OPEN YOUR IRIS APP</a></h3>'))
    
except Exception as e:
    print(f"⚠️ Could not get public URL automatically: {e}")
    print("\n📱 **MANUAL STEPS:**")
    print("1. Look at the RIGHT side of Colab")
    print("2. Find the 🔗 icon (between 💬 and 🎮)")
    print("3. Click it → Select 'Preview on port 8501'")
    print("\nOR try this direct link:")
    print("https://xxxxxxxx-8501-colab.googleusercontent.com")

print("\n⏳ Starting server in background...")
print("💡 **Keep this cell running!** Don't press STOP unless you want to close the app.")

# Start Streamlit in background
import subprocess, threading
import sys

def run_streamlit():
    process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', 'iris_app.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.serverAddress', 'localhost',
        '--theme.base', 'light'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print output
    for line in process.stdout:
        print(f"[Streamlit] {line}", end='')
    for line in process.stderr:
        print(f"[Streamlit ERROR] {line}", end='')

# Run in thread
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

print("✅ Server started! Waiting 10 seconds for it to be ready...")
import time
time.sleep(10)

print("\n" + "="*70)
print("📱 **APP STATUS:** Running on port 8501")
print("💡 If the link above doesn't work:")
print("1. Refresh your browser")
print("2. Try clicking the 🔗 icon on the right")
print("3. OR copy-paste the URL manually")
print("="*70)

# Keep alive
try:
    while True:
        time.sleep(60)
        print("⏱️ App still running... (Click STOP ⏹️ to exit)")
except KeyboardInterrupt:
    print("\n🛑 App stopped by user")
