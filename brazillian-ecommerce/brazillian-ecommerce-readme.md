# Brazillian Olist Ecommerce Project
### Using Ecommerce Data for Analyzing Buyer Retention and Seller Acquisition
- Report to Business Stakeholders: [Go to Business Insights](#bulb-business-insights)
- Report to Tech Stakeholders: [Go to Data Processing](#toolbox-data-processing)


# :bulb: BUSINESS INSIGHTS
[Scroll down for business recommendations](#bulb-recommendations)
## Dashboard Peek
:link: [Go to Tableau Dashboard](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/viz_brazillian_ecommerce/CUSTOMERRETENTION)


## Northstar Metrics: Average Order Value
- AVO is a metrics used to measure how much a customer spent on an ecommerce platform.

## Insights from Visualization
#### *VIEW 1* - Overview of Olist
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_1.png)  |
|------------------------|
##### VIEW 1 INSIGHTS
*Context*
- As this is an ecommerce platform, their customers include two main buckets: one is the sellers and the other one is the buyers. From the main dashboard, there are two main trends. First is that the acquistion of customer grew strongly throughout the examined period. Sellers growth is comparatively slower and plateaud around 380 new sellers in the latter half.
- The typical amount that customer pay is around R$158 and average listings of each seller on the platform is 38 items, which is quite a large product portfolio for small-and-medium sized businesses.
- What is most noticeable is the extremely slow customer retention rate and seller retention rate. We will now move on exploring what could potentially cause the slow retention rate.


#### *VIEW 2* - Low Customer Retention Inspection
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_2.png)  |
|------------------------|
##### VIEW 2 INSIGHTS
- Before going into the strategy to improve our customer retention, I first explored the motivations behind increasing the number of return and repeat customers. The reason is that repeat customers spend 35% higher and 10% higher than one-time buyers. 
- The detailed cohort table shows monthly retention consistently in the 0.1% to 0.5% range. New customers in any given month almost never return in subsequent months.


#### *VIEW 3* - Finding Patterns in Items Bought and Payment Methods
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_3.png)  |
|------------------------|
##### VIEW 3 INSIGHTS
To understand why buyers do not have interest to stay and make frequent purchases on our platform. I decided to take a look back at their customer journy, from their online purchase experience, to post-purchase experience like receiving products and writing reviews.



#### *VIEW 4* - Exploring Isssues Related to Delivery
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_4.png)  |
|------------------------|
##### VIEW 4 INSIGHTS
- Our customers are not high-income customers but rather low-to-middle income customer who make lots of purchases for low-priced items and they only bought a small quantity of them. Most payments are via credit card. The largest consumed product categories include health and beauty, bed and bath, sports and leisure, watches, and gift.
- My assumption is that a good customer experience, reflected through a good review score, would be the main motivation for customers to make subsequent purchases.
- My hypothesis about the delivery and product review is true. Faster delivery leads to higher review scores, which could subsequently ends in a revisit.
- My recommendations is that Olist should reduce the 9.2 days average shipping time and the 2.8 days average order preparation time to improve the customer review score. 


#### *VIEW 5* - Inspecting Seller Retention
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_5.png)  |
|------------------------|
##### VIEW 5 INSIGHTS
- Most new sellers are churning immediately, validating the overall 10.53% low seller retention rate. This indicates a severe problem with the seller onboarding experience, platform tools, or commission structure.


#### *VIEW 6* - Seller Acquisition and Conversion 
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_6.png)  |
|------------------------|
##### VIEW 6 INSIGHTS
- Assuming that seller retention also relies on how they were first contacted, I dived deeper into the CRM activities of the company.
- I built this dashboard specifically for sales manager to know who to engage with when dealing with different kinds of leads. Some SDRs and SRs have superior performance than the other members in their teams. The sales manager can also reward exmployees with high KPIs.

#### *VIEW 7* - Big Picture of Olist Seller and Buyer Network
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_7.png)  |
|------------------------|
##### VIEW 7 INSIGHTS
- Geographic Distribution: Both customers and sellers are heavily concentrated in the Southeast region of Brazil, particularly the states of São Paulo (SP), Minas Gerais (MG), and Rio de Janeiro (RJ).
- Warehouse Recommendation: The heatmap confirms this concentration, showing the highest order density (darker squares) in the SP, RJ, and MG regions.
- Recommendation: To reduce the 9.2-day average shipping time, the next regional warehouse should be located in the Southeast Region (e.g., São Paulo or Rio de Janeiro) to serve the high demand of buyers and allow the sellers to load their products to our warehouses faster.

## :bulb: Recommendations
- Improving CX: reducing the number of waiting days to have their products delivered. Building a new warehouse or relocating a current warehouse location to the central areas.
- Due to the low number customers and the AOV is small, sellers are leaving the platform to pursue their businesses somewhere else. My recommendation is that we should give them some insights that we have collected about their customers and provide strategic recommendations for them, including cross-selling and offering bargains.


# :toolbox: DATA PROCESSING
## Tech Stack
- SQL (Snowflake)
- [Tableau (Visualization)](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/viz_brazillian_ecommerce/CUSTOMERRETENTION)

## 1. Data Source Introduction

### 1a. Data Source Link
The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

- [Olist Real System Data Collected and Anonymized](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data?select=olist_order_payments_dataset.csv)
- [Olist Real Marketing Funnel Data](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist)
- Brainstorming on how to extract insights from those columns using [Go to Excel](https://docs.google.com/spreadsheets/d/1h2Xs7e2DcjXTEfsc9LuBtVi2JQAp4Vvw4mY1suWDA3Q/edit?gid=1664708578#gid=1664708578)

## 2. Snowflake - Data Cleaning
 - Files used for cleaning (downloaded from Snowflake workspace)
	 - [load_and_clean_ecommerce](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/brazillian-ecommerce/sql/load_and_clean_ecommerce.sql)
   - [interact_ecommerce](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/brazillian-ecommerce/sql/interact_ecommerce.sql)

## Technical Challenges
- Data was collected from more than 5 sources and merging the tables was a real challenge for me. I have to make sure the column names were consistent across the tables.
- Regarding the missing values, they are not missing by system log errors but occur because the process has not happened yet. Interpreting that the columns are parts of a process allowed me to go ahead leaving these null values or replacing it with 0 to enable date calculation like shipping time, order preparation days.
- As this is a large datasets with more than 20 dimensions, deciding what to use for the analysis was difficult. I abandoned some attributes related to the product such as product height or product size. At the same time, it provided lots of rooms for engineering new features like classifying an order or classifying a customer as repeat or return customers.
- The key to join table 'customers' and table 'orders' is customer_id but it should not be used for analysis because it is just a joining key. The true customer identifier column is customer_unique_id and after finish joining the table, I dropped the 'customer_id' column and changed 'customer_unique_id' column to 'customer_id' to avoid any confusion.
- I am not yet able to deal with handling text data and natural language processing related task so I also dropped columns review_comment_title and review_comment_message which could provide lots of insights regarding customer experience. 

