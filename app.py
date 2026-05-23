import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supermarket Sales Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { font-size: 18px; }
    h1 { font-size: 38px !important; }
    h3 { font-size: 24px !important; }
    .stMetric {background-color: #1c2630;border: 1px solid #3a4753;padding: 16px;border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data #load the data on first open and then store it in the cache to save on loading time for the big DF
@st.cache_data
def load_and_prepare():
    df_transaction = pd.read_csv(
        "transaction_data.csv.gz",
        usecols=["household_key", "BASKET_ID", "PRODUCT_ID",
                 "QUANTITY", "SALES_VALUE", "WEEK_NO"],
    )
    df_product = pd.read_csv(
        "product.csv.gz",
        usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"],
    )
    df = df_transaction[(df_transaction["QUANTITY"] > 0) & (df_transaction["SALES_VALUE"] > 0)]
    df = df.merge(df_product[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]], on="PRODUCT_ID")
    return df

df = load_and_prepare() #prep data for use - one slow load to start and then available for use

# Introduction Area

st.title("Retail Sales Overview Dashboard")

st.markdown("""This dashboard is a summary of two years of supermarket shopping data from **~2,500 US households**, sourced from Dunnhumby's "The Complete Journey". 

Use the dropdown below to explore data for each store department, or leave it on **All Departments** to see the full picture.""")


# Department selection dropdown

dept_options = ["All Departments"] + sorted(df["DEPARTMENT"].unique().tolist()) #full df or a sorted list of all depts
selected_dept = st.selectbox("Select a department:",dept_options,help="Choose a department to filter the headline figures and charts below.")

if selected_dept == "All Departments":
    sub = df #if all dept chosen then graphs df is full df
else:
    sub = df[df["DEPARTMENT"] == selected_dept] #filter df for graphs to just the dept chosen

# Headline figures 
st.divider() 

st.markdown(f"### **{selected_dept}** - Headline figures")

st.markdown("""These figures give a quick, high-level summary of the department you have selected: the total amount spent, the number of separate shopping trips, how many different households shopped, and how many product categories were involved.""")

col1, col2, col3, col4 = st.columns(4) #list figures side by side as columns 
col1.metric("Total Sales",f"${sub['SALES_VALUE'].sum():,.0f}")
col2.metric("Baskets",f"{sub['BASKET_ID'].nunique():,}") 
col3.metric("Households",f"{sub['household_key'].nunique():,}") #
col4.metric("Commodity Categories",f"{sub['COMMODITY_DESC'].nunique():,}")


#ML Suitablity Section
st.divider() 

st.markdown("### Why this dataset is suitable for Machine Learning")

st.markdown("""Machine learning works by finding patterns in past shopping behaviour and the two figures below show why this data is well suited for this. Households return many times, so we can learn their habits, and with each basket holding several items, we can see which products tend to be bought together.""")

avg_baskets_per_household = sub.groupby("household_key")["BASKET_ID"].nunique().mean() #avg number of unique baskets per household 
avg_items_per_basket = sub.groupby("BASKET_ID").size().mean() #avg basket size per household

col1, col2 = st.columns(2) 
col1.metric(
    "Average shopping trips per household",
    f"{avg_baskets_per_household:.1f}",
    help="Recommender systems need shoppers with repeat behaviour. A high average means we observe genuine customer histories, not one-off visits."
)
col2.metric(
    "Average items per basket",
    f"{avg_items_per_basket:.1f}",
    help="Market basket analysis needs baskets containing multiple items so co-occurrence patterns can be detected."
)


# Top Dept commodities by sales bar chart 
st.divider() 

st.markdown(f"### Top 10 Commodities by sales in **{selected_dept}**")

st.markdown("""This chart shows the ten product categories that brought in the most sales for the selected department. The longer the bar, the higher the total sales for that category.""")

chart_data = sub.groupby("COMMODITY_DESC")["SALES_VALUE"].sum().nlargest(10).reset_index() #taking 10 commoditys in dept by largest sales val 

fig = px.bar(chart_data,x="SALES_VALUE", y="COMMODITY_DESC",orientation="h",labels={"SALES_VALUE": "Total sales ($)", "COMMODITY_DESC": ""})
fig.update_layout(yaxis={"categoryorder": "total ascending"}, font=dict(size=16))
st.plotly_chart(fig, use_container_width=True)

# department weekly sales trend chart 
st.divider() 

st.markdown(f"### **{selected_dept}** - Weekly sales trend")

st.markdown("""This chart shows how total sales rose and fell week by week across the two-year period, making it easy to spot busier and quieter times.""")

trend = sub.groupby("WEEK_NO", as_index=False)["SALES_VALUE"].sum().sort_values("WEEK_NO")

fig = px.line(trend,x="WEEK_NO", y="SALES_VALUE",labels={"WEEK_NO": "Week: ", "SALES_VALUE": "Total sales ($)"})
fig.update_layout(font=dict(size=16))
fig.update_traces(line=dict(width=3)) 
st.plotly_chart(fig, use_container_width=True)


# Department tree map of Sub-Commodities
st.divider()
st.markdown(f"### Sales breakdown — commodities within {selected_dept}")

st.markdown("""The below Tree graph shows a breakdown of the most popular items by department. The percentage denotes out of the full department sales, what % are from that particular item.   
            Hover your mouse over each item in order to get total sales values for the item.""")

breakdown = sub.groupby(["DEPARTMENT", "COMMODITY_DESC"], as_index=False)["SALES_VALUE"].sum()

fig = px.treemap(breakdown,path=["DEPARTMENT", "COMMODITY_DESC"],values="SALES_VALUE",)
fig.update_layout(font=dict(size=14), margin=dict(t=10, l=10, r=10, b=10), height=600)
fig.update_traces(textinfo="label+percent parent")
st.plotly_chart(fig, use_container_width=True)
