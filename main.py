import numpy as np
import datetime as dt
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('bigmart_model')

# Current year for the Outlet Age calculation
current_year = dt.datetime.today().year

# Streamlit app title
st.title("Sales Prediction using Machine Learning")

# Input fields
st.subheader("Enter the following details:")

item_mrp = st.number_input("Item Price", min_value=0.0, format="%.2f")
outlet_identifier = st.selectbox("Outlet Identifier", 
                                ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 
                                'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
outlet_size = st.selectbox("Outlet Size", ['High', 'Medium', 'Small'])
outlet_type = st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=current_year)

outlet_age = current_year - outlet_establishment_year

# Convert values to numeric
outlet_identifier_map = {
    "OUT010": 0, "OUT013": 1, "OUT017": 2, "OUT018": 3, "OUT019": 4, 
    "OUT027": 5, "OUT035": 6, "OUT045": 7, "OUT046": 8, "OUT049": 9
}

outlet_size_map = {"High": 0, "Medium": 1, "Small": 2}

outlet_type_map = {
    "Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3
}

# Mapping the inputs to numeric
p2 = outlet_identifier_map[outlet_identifier]
p3 = outlet_size_map[outlet_size]
p4 = outlet_type_map[outlet_type]
p5 = outlet_age

# Prediction button
if st.button("Predict"):
    prediction_input = np.array([[item_mrp, p2, p3, p4, p5]])
    
    result = model.predict(prediction_input)
    
    st.subheader("Predicted Sales Amount Range")
    st.write(f"The sales amount is between: {float(result) - 714.42:.2f} and {float(result) + 714.42:.2f}")
    st.write(f"Predicted Sales Amount: {float(result):.2f}")
    
    # Plotting the graph
    fig, ax = plt.subplots()
    ax.bar(["Predicted Sales"], [float(result)], color='blue', alpha=0.6, label="Predicted Sales")
    ax.bar(["Predicted Sales"], [float(result) - 714.42], color='orange', alpha=0.6, label="Lower Bound")
    ax.bar(["Predicted Sales"], [float(result) + 714.42], color='red', alpha=0.6, label="Upper Bound")
    
    ax.set_title("Predicted Sales Amount with Range")
    ax.set_ylabel("Sales Amount")
    ax.legend()
    
    st.pyplot(fig)
