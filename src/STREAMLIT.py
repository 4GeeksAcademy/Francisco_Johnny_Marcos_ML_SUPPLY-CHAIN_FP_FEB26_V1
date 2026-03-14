import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import pydeck as pdk
import plotly.graph_objects as go
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Supply Chain ML Dashboard",
    layout="wide",
    page_icon="📦"
)

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #CCA465;
}
h1 {
    text-align:center;
    color:#CCA465;
}
.block-container {
    padding-top:2rem;
}
[data-testid="stSidebar"] {
    background-color:#CCA465;
}
[data-testid="stSidebar"] label {
    color:white;
}
.table-style {
    background-color:white;
    border-radius:10px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_model():
    with open("models/supervised_model_final_boost.pkl", "rb") as f:
        model = joblib.load(f)
    return model
def load_scaler():
    with open("models/scaler_WITHOUT_outliers.pkl", "rb") as f:
        model = joblib.load(f)
    return model

def load_model2():
    with open("models/unsupervised_kmeans_final.pkl", "rb") as f:
        model2 = joblib.load(f)
    return model2

model = load_model()
model2= load_model2()
scaler=load_scaler()
# -------------------------
# LOAD MAPPINGS
# -------------------------
with open("data/interim/category_mappings.json") as f:
    mappings = json.load(f)

# -------------------------
# COUNTRY COORDINATES
# -------------------------
with open("src/country_coords.json", "r", encoding="utf-8") as f:
    country_coords = json.load(f)

with open("src/city_to_countries.json", "r", encoding="utf-8") as f:
    country_to_cities = json.load(f)
# -------------------------
# HELPER FUNCTION
# -------------------------
def select_from_mapping(label, mapping):
    if isinstance(mapping, dict):
        options = list(mapping.keys())
        selected = st.sidebar.selectbox(label, options)
        value = mapping[selected]
    elif isinstance(mapping, list):
        options = mapping
        selected = st.sidebar.selectbox(label, options)
        value = options.index(selected)
    else:
        st.error(f"Mapping format not supported for {label}")
        return None, None
    return selected, value

def select_from_list(label, options):
    selected = st.sidebar.selectbox(label, options)
    selected_num = options.index(selected)
    return selected, selected_num

# -------------------------
# TITLE
# -------------------------
st.title("📦 DataCo Supply Chain Risk Predictor")
st.markdown(
    "<center>Predict delivery delays and Identify logistic risk clusters in real-time.</center>",
    unsafe_allow_html=True
)

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("📥 Input Order Data")
st.sidebar.header("Step 1: Order Logistics")
# 1. País primero
Customer_Country, Customer_Country_num = select_from_mapping("Origin Country", mappings["Customer_Country"])

# 2. Ciudades disponibles según país
available_cities2 = country_to_cities.get(Customer_Country, [])

# 3. Filtrar contra las ciudades reales del mapping
filtered_order_cities2 = [
    city for city in available_cities2
    if city in mappings["Customer_City"]
]

# 4. Selector de ciudad
if filtered_order_cities2:
    Customer_City, Customer_City_num = select_from_list("Origin City", filtered_order_cities2)
else:
    st.sidebar.warning(f"No hay ciudades mapeadas para {Customer_Country}. Se muestran todas.")
    Customer_City, Customer_City_num = select_from_list("Origin City", mappings["Customer_City"])



shipping_day, shipping_day_num = select_from_mapping("Shipment Day", mappings["shipping_day"])

shipping_mode_map = {0:"Standard Class",1:"Second Class",2:"First Class",3:"Same Day"}
Shipping_Mode = st.sidebar.selectbox("Shipping Mode", list(shipping_mode_map.values()))
Shipping_Mode_num = [k for k,v in shipping_mode_map.items() if v == Shipping_Mode][0]


st.sidebar.header("Step 2: Product & Payment")
Category_Name, Category_Name_num = select_from_mapping("Category", mappings["Category_Name"])
type_map = {0:"DEBIT",1:"TRANSFER",2:"CASH",3:"PAYMENT",4:"CREDIT"}
Type = st.sidebar.selectbox("Payment Type", list(type_map.values()))
Type_num = [k for k,v in type_map.items() if v == Type][0]

st.sidebar.header("Step 3: Constraints")
Days_for_shipment_scheduled = st.sidebar.slider("Scheduled Days",1,10,3)
Price_Per_Unit = st.sidebar.number_input("Unit Price ($)", value=150.0)
Benefit_per_order = st.sidebar.number_input("Expected Benefit ($)", value=50.0)

st.sidebar.header("Step 4: Shipping info")
# 1. País primero
Order_Country, Order_Country_num = select_from_mapping("Order Country", mappings["Order_Country"])

# 2. Ciudades disponibles según país
available_cities = country_to_cities.get(Order_Country, [])

# 3. Filtrar contra las ciudades reales del mapping
filtered_order_cities = [
    city for city in available_cities
    if city in mappings["Order_City"]
]

# 4. Selector de ciudad
if filtered_order_cities:
    Order_City, Order_City_num = select_from_list("Order City", filtered_order_cities)
else:
    st.sidebar.warning(f"No hay ciudades mapeadas para {Order_Country}. Se muestran todas.")
    Order_City, Order_City_num = select_from_list("Order City", mappings["Order_City"])


Order_Status, Order_Status_num = select_from_mapping("Order Status", mappings["Order_Status"])


#---------

# PREDICTION  

#-----------

predict_button = st.sidebar.button("🚀 Run Prediction")

predictors = [
    'Days_for_shipment_scheduled', 'Benefit_per_order', 'Order_Item_Discount', 
    'Order_Item_Discount_Rate', 'Order_Item_Profit_Ratio', 'Order_Item_Quantity', 
    'Type_num', 'Category_Name_num', 'Customer_City_num', 'Customer_Country_num', 
    'Customer_Segment_num', 'Customer_State_num', 'Department_Name_num', 
    'Order_City_num', 'Order_Country_num', 'Order_State_num', 'Order_Status_num', 
    'Shipping_Mode_num', 'Customer_Zipcode_num', 'shipping_day_num', 
    'shipping_month_num', 'Price_Per_Unit', 'Logistics_Corridor_ID'
]
input_data2 = pd.DataFrame(np.zeros((1, len(predictors))), columns= predictors)
input_data2['Days_for_shipment_scheduled'] = Days_for_shipment_scheduled
input_data2['Benefit_per_order'] = Benefit_per_order
input_data2['Price_Per_Unit'] = Price_Per_Unit
input_data2['Shipping_Mode_num'] = Shipping_Mode_num
input_data2['shipping_day_num'] = shipping_day_num
input_data2['Type_num'] = Type_num
input_data2['Order_City_num'] = Order_City_num
input_data2['Category_Name_num'] = Category_Name_num
input_data2['Order_Status_num'] = Order_Status_num
cluster_status = {
    0: "🟡 Moderate Risk (Standard)",
    1: "🟢 Low Risk (Optimal)",
    2: "🔴 Critical Risk (Impossible Schedule)"
}
input_scaled = scaler.transform(input_data2)

if predict_button:
    # Supervised prediction
    prediction = model.predict(input_data2)[0]
    prob = model.predict_proba(input_data2)[0][1] # Probability of delay

    # Unsupervised cluster assigment
    cluster = model2.predict(input_scaled)[0]
    readable_cluster = cluster_status.get(cluster, "Unknown Cluster")

    # 4. Interactive UI display
    st.divider()
    # Bloque de probabilidad y estado
    with st.container():
        col1, col2 = st.columns([1, 2])  # Col2 más ancho
    with col1:
        st.metric("Late Risk Probability", f"{prob * 100:.1f}%")
    with col2:
        if prediction == 1:
            st.error("Status: LATE EXPECTED")
        else:
            st.success("Status: ON TIME")

    # Bloque de perfil logístico
    
    if prediction == 0:
        st.info(
            "**Congrats**: The current scheduling window is optimal. This order is highly likely to meet its deadline.")

    else:
        with st.container():
            st.metric("Logistic Profile", readable_cluster)
        if cluster == 2:
            st.warning(
                "**Strategic Insight**: This order is being promised too fast for our current logistics capacity. Recommend increasing the scheduled days to at least 3."
            )
            st.subheader("Strategic Recommendation")
            st.error("Action Required: Reschedule Order")
            st.write(f"The current promise of **{scheduled_days} day(s)** is physically impossible for our current logistics to {selected_city}.")
        
            # Calculate the 'Safe' target
            suggested_days = 4 # Based on Cluster 1 average
            additional_days = suggested_days - scheduled_days
        
            st.info(f"**To move this to 'Low Risk' (Cluster 1):** Increase the 'Scheduled Days' to **{suggested_days}**. "
                f"This adds {additional_days} day(s) to the customer promise but ensures an 85%+ on-time delivery rate.")
            
        elif cluster == 1:
            st.info(
             "**Optimization Tip**: This profile is highly efficient. Continue using these parameters for this route."
        )
        st.subheader("Strategic Recommendation")
        st.warning("Action Recommended: Review Buffer")
        st.write("This order is in the 'Moderate' zone. Adding **1 extra day** to the schedule would likely shift this into the 'Low Risk' green zone.")
        
# -------------------------
# MAP VISUALIZATION
# -------------------------
st.subheader("🌍 Shipping Route")
if Customer_Country in country_coords and Order_Country in country_coords:
    origin = country_coords[Customer_Country]
    destination = country_coords[Order_Country]

    map_data = pd.DataFrame({"lat":[origin[1], destination[1]], "lon":[origin[0], destination[0]]})
    arc = pd.DataFrame({
        "start_lon":[origin[0]], "start_lat":[origin[1]],
        "end_lon":[destination[0]], "end_lat":[destination[1]]
    })

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position='[lon, lat]',
        get_radius=250000,
        get_fill_color='[0,102,255]'
    )
    layer_arc = pdk.Layer(
        "ArcLayer",
        data=arc,
        get_source_position='[start_lon,start_lat]',
        get_target_position='[end_lon,end_lat]',
        get_source_color=[0,150,255],
        get_target_color=[255,100,0],
        get_width=6
    )
    view_state = pdk.ViewState(
        latitude=(origin[1]+destination[1])/2,
        longitude=(origin[0]+destination[0])/2,
        zoom=1
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer_arc, layer_points],
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        )
    )

