import streamlit as st
import pickle
import numpy as np

# Load saved files
with open("kmeans_model.pkl", "rb") as file:
    kmeans_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("recommendation_dict.pkl", "rb") as file:
    recommendation_dict = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation and Product Recommendation System")

# Sidebar
menu = st.sidebar.radio(
    "Select Module",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# Home Page
if menu == "Home":
    st.header("Project Overview")

    st.write("""
    Shopper Spectrum is an e-commerce analytics project that performs customer segmentation
    using RFM analysis and K-Means clustering. It also recommends similar products using
    item-based collaborative filtering.
    """)

    st.markdown("""
    ### Features
    ● Customer segmentation using Recency, Frequency, and Monetary values  
    ● Product recommendation using collaborative filtering  
    ● Real-time prediction through Streamlit interface  
    ● Useful for targeted marketing, customer retention, and product personalization  
    """)

# Customer Segmentation Page
elif menu == "Customer Segmentation":
    st.header("🎯 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)

        cluster = kmeans_model.predict(input_scaled)[0]

        # Rule-based segment interpretation
        if recency <= 50 and frequency >= 10 and monetary >= 1000:
            segment = "High-Value"
            message = "This customer is highly valuable, purchases frequently, and spends more."
        elif frequency >= 5 and monetary >= 500:
            segment = "Regular"
            message = "This customer purchases steadily and has moderate spending behavior."
        elif recency >= 150 and frequency <= 2 and monetary <= 300:
            segment = "At-Risk"
            message = "This customer has not purchased recently and may require retention campaigns."
        else:
            segment = "Occasional"
            message = "This customer purchases occasionally and can be targeted with personalized offers."

        st.success(f"Predicted Cluster: {cluster}")
        st.info(f"Customer Segment: {segment}")
        st.write(message)

# Product Recommendation Page
elif menu == "Product Recommendation":
    st.header("🛍️ Product Recommendation")

    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:
            product_name_upper = product_name.upper()

            matched_product = None

            for product in recommendation_dict.keys():
                if product_name_upper in product.upper():
                    matched_product = product
                    break

            if matched_product is None:
                st.error("Product not found. Please try another product name.")
            else:
                st.success(f"Recommendations for: {matched_product}")

                recommendations = recommendation_dict[matched_product]

                for i, product in enumerate(recommendations, start=1):
                    st.write(f"{i}. {product}")