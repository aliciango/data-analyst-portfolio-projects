# (wip) Brazillian Ecommerce Project
- Report to Business Stakeholders: [Go to Business Insights](#bulb-business-insights)
- Report to Tech Stakeholders: [Go to Data Processing](#toolbox-data-processing)


# :bulb: BUSINESS INSIGHTS
## Dashboard Peek
:link: [Go to Tableau Dashboard](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/viz_brazillian_ecommerce/CUSTOMERRETENTION)


## Northstar Metrics: Page Value
At the end of the day, the goal of this dashboard is finding ways to increase the page value. Page value is calculated by dividing the dollar amount of customer purchases by the number of pages that they visit during a session. 

## Insights from Visualization
#### *VIEW 1* - Overview of User Activities during the last 12 months
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_1_overview.png)  |
|------------------------|
##### VIEW 1 INSIGHTS


#### *VIEW 2* - Inspecting how behavior of converters and non-converters are different
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_2_cust_retention.png)  |
|------------------------|
##### VIEW 2 INSIGHTS



#### *VIEW 3* - Finding root causes for the behavior discrepancy between converters and non-converters
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_3_cust_delivery_hyp.png)  |
|------------------------|
##### VIEW 3 INSIGHTS



#### *VIEW 4* - Finding root causes for the behavior discrepancy between converters and non-converters
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_4_cust_product_hyp.png)  |
|------------------------|
##### VIEW 4 INSIGHTS


#### *VIEW 5* - Finding root causes for the behavior discrepancy between converters and non-converters
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_5_seller_retention.png)  |
|------------------------|
##### VIEW 5 INSIGHTS


#### *VIEW 6* - Finding root causes for the behavior discrepancy between converters and non-converters
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_6_seller_hannel.png)  |
|------------------------|
##### VIEW 6 INSIGHTS


#### *VIEW 7* - Finding root causes for the behavior discrepancy between converters and non-converters
| ![image alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/ecommerce_7_seller_buyer_network.png)  |
|------------------------|
##### VIEW 7 INSIGHTS



# :toolbox: DATA PROCESSING
## Tech Stack
- SQL (Snowflake)
- [Tableau (Visualization)](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/viz_brazillian_ecommerce/CUSTOMERRETENTION)

## 1. Data Source Introduction

### 1a. Data Source Link
- [Olist Real System Data Collected and Anonymized](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data?select=olist_order_payments_dataset.csv)
- [Olist Real Marketing Funnel Data](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist)
- Brainstorming on how to extract insights from those columns using [Go to Excel](https://docs.google.com/spreadsheets/d/1h2Xs7e2DcjXTEfsc9LuBtVi2JQAp4Vvw4mY1suWDA3Q/edit?gid=1664708578#gid=1664708578)

## 2. Snowflake - Data Cleaning
 - Files used for cleaning (downloaded from Snowflake workspace)
	 - [load_and_clean_ecommerce](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/brazillian-ecommerce/sql/load_and_clean_ecommerce.sql)
   - [interact_ecommerce](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/brazillian-ecommerce/sql/interact_ecommerce.sql)

## Technical Challenges
- data came from different sources,

