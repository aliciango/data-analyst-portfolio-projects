# Unicorn Companies Project
- Report to Business Stakeholders: [Go to Business Insights](#bulb-business-insights)
- Report to Tech Stakeholders: [Go to Data Processing](#toolbox-data-processing)


# :bulb: BUSINESS INSIGHTS
## Dashboard Peek
:link: [Go to Tableau Dashboard](https://public.tableau.com/views/FINISHEDDASHBOARD/DEALSOURCINGDB?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d936e06e34322df8660598fec4c685f3698567cf/assets/unicorn_tableau.png)
@[image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d936e06e34322df8660598fec4c685f3698567cf/assets/unicorn_tableau2.png)



![Logo](<!-- Add your logo image link here -->)

## Client Background

**[Client Name]** is a [describe the business or organization — e.g., "global retail company focused on digital commerce"].

This report provides an analytical overview of [client's or project's] recent performance trends, challenges, and opportunities.

The analysis focuses on key performance indicators, data insights, and strategic recommendations to support decision-making and operational improvements.

## Northstar Metrics

- [Metric 1] – e.g., Revenue growth trends and seasonal performance.
- [Metric 2] – e.g., Product performance and category insights.
- [Metric 3] – e.g., Customer engagement or loyalty program evaluation.
- [Metric 4] – e.g., Regional sales comparison and market penetration.

## Executive Summary

1. **[Key Trend or Highlight 1]**
   - Briefly summarize the most significant finding or event.
   - Include the general outcome or trend (e.g., steady growth, seasonal dip).

2. **[Key Trend or Highlight 2]**
   - Describe the cause or possible contributing factors.
   - Mention implications or impact areas (e.g., market shift, demand change).

3. **[Additional Insight or Observation]**
   - Highlight any repeating pattern or cyclical trend.
   - Briefly touch on data anomalies or outliers worth further exploration.

4. **[Key Takeaways & Recommendations]**
   - List actionable recommendations based on the analysis.
   - Summarize potential strategies or next steps.


## Insights Deep-Dive

### Sales Trend

![Sales Trend](<!-- Add sales trend chart -->) ![AOV Chart](<!-- Add AOV chart -->) ![Order Count](<!-- Add order count chart -->)

Briefly describe sales trends — e.g., periods of peak sales, downturns, or anomalies.

---

### Product Performance

![Product Performance](<!-- Add product performance chart -->)

Summarize product category performance and identify top or underperforming items.

---

### Loyalty Program Insights
[Loyalty Metric 1](<!-- Add chart -->)

[Loyalty Metric 2](<!-- Add chart -->)

Explain loyalty member performance differences and customer engagement patterns.

---

### Refund Rates
![Refund Analysis](<!-- Add refund chart -->)

- Summarize key findings on refund frequency and product reliability.
- Note which product types or categories have higher return rates.
- Provide 1–2 possible improvement suggestions.

---

### Regional Analysis

![Regional Analysis](<!-- Add regional chart -->)

Describe key performance differences between regions and potential growth opportunities.

---

## Recommendations

Based on the insights, here are the proposed strategic actions:

### Sales
- Implement strategies to stabilize sales across low-performing months.
- Focus on campaigns that leverage historical high-growth periods.

### Products
- Increase inventory or marketing focus on high-performing product categories.
- Reassess low-performing SKUs for discontinuation or repositioning.

### Loyalty Program
- Encourage membership through targeted incentives and personalized offers.
- Use loyalty data to segment and retain high-value customers.

### Refunds & Quality
- Address quality issues in products with high return rates.
- Improve post-purchase support to reduce unnecessary returns.

### Region
- Address quality issues in products with high return rates.
- Improve post-purchase support to reduce unnecessary returns.



# :toolbox: DATA PROCESSING

## Tech Stack
- Python (pandas and numpy for data cleaning)
- [Snowflake/SQL](#snowflake) (database design, merge tables, quick querying)
- [Tableau (Visualization)](https://public.tableau.com/views/FINISHEDDASHBOARD/DEALSOURCINGDB?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 1. Data Source Introduction

### 1a. Data Source Link
- [2024 unicorn data](https://www.kaggle.com/datasets/sashakorovkina/2024-unicorn-and-emerging-unicorn-companies?resource=download&select=unicorn_companies.csv)
- [2023 unicorn data](https://www.kaggle.com/datasets/ritwikb3/unicorn-companies)
- [2022 unicorn data](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies)

3 CSV files that collect information about private companies with a value of over $1 billion as of March 2022, 2023, and 2024.

### 1b. Variable Description
- Dataset 1:


| *unicorn_companies_2022.csv* | Description | Data Type |
|-------------|-------------|-----------|
| company | The name of the company. | Text |
| city | The headquarters city of the company. | Text |
| country | The headquarters country/region of the company. | Text |
| city/region | The headquarters continent of the company. | Categorical |
| industry | The industry or sector the company operates in. | Categorical |
| select_investors | A list of key investors in the company. | Text |
| funding | The total funding raised by the company (as a string). | Text |
| valuation | The company's valuation in billions of dollars. | Numerical |
| year_founded | The year the company was founded. | Numerical |
| date_joined | The date the company reached unicorn status ($1B+ valuation). | Temporal |
| funding_currency |	**data cleaned column**, split from the 'Funding' column. Categories: B (billion), M (million) | Categorical
| funding_value	| **data cleaned column**, split from the 'Funding' column |	Numerical
| years_to_unicorn | **feature engineering**, ***= year_founded - date_joined*** | Categorical |

 - Dataset 2:

| *unicorn_companies_2023.csv* | Description | Data Type |
|-------------|-------------|-----------|
| company | The official name of the unicorn company. | Text |
| valuation | The company's valuation in Billions of US Dollars ($B). | Numerical |
| select_investors | A sample list of key investors or venture capital firms. | Text |
| city | The city where the company is headquartered. | Categorical |
| date_joined | The date the company first reached a $1 billion valuation. | Temporal |
| country/region | The country where the company is headquartered. | Categorical |
| industry | The primary industry or sector of the company. | Categorical |
| date_joined_cleaned| The primary industry or sector of the company. | Categorical |

 - Dataset 3:

| *unicorn_companies_2024.csv* | Description |
|-------------|-------------|
| country | Country name |
| country/region | Region |
| select_investors |  |
| company_link |  |
| img_src |  |
| company |  |
| valuation_raw |  |
| funding_raw |  |
| valuation_currency | **data cleaned column**, split from the valuation_raw. Value contains: B |
| valuation | **data cleaned column**, numeric values |
| company_cleaned | **data cleaned column**, uppercase of company name and without space |
## 2. Python - Data Cleaning
 - Files used for cleaning 
	 - [eda.py](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d2fa679ef0a0d917d1ef82b757c2009d79210d3d/unicorn-companies/eda.py)
	 - [eda_23.py](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d2fa679ef0a0d917d1ef82b757c2009d79210d3d/unicorn-companies/eda_23.py)
	 - [eda_24.py](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d2fa679ef0a0d917d1ef82b757c2009d79210d3d/unicorn-companies/eda_24.py)
## 3. Snowflake (SQL) - Database Design and Preparing Data for Visualization
[Snowflake Star & Query Testing](https://app.snowflake.com/axnfple/uhb18311/w31UKE6fN8gN#query)
[Snowflake One Big Table](https://app.snowflake.com/axnfple/uhb18311/w413ag1rvuKi#query)
## SNOWFLAKE
Now that we already have three cleaned datasets, it is time to combine them. Actually cleaning is an iterative process, so I discovered a few more data quality-related issues while executing queries in snowflake. Therefore, I created the file 
joining_prep.py to do any additional cleaning task. I also used excel to record the current columns that I have in each csv file and document how I am going to use the columns for joining these separate tables from 2022, 2023 and 2024.

| Column Name | 2022 | 2023 | 2024 |
|-------------|------|------|------|
| COMPANY_CLEANED | O | O | O |
| SELECT_INVESTORS | O | O | O |
| VALUATION | O | O | O |
| COUNTRY | O | O | O |
| FUNDING | O | X | O |
| CITY | O | O | X |
| YEARS_TO_UNICORN | O | O | X |
| DATE_JOINED | O | O | X |
| INDUSTRY | O | O | X |
| FUNDING_CURRENCY | O | X | X |
| YEAR_FOUNDED | O | X | X |
| IMG_SRC | X | X | O |
| COMPANY_LINK | X | X | O |
| VALUATION_CURRENCY | X | X | O |

As the data is just crawled from the internet, it is really messy. For the sake of updating and adding the data in the future, I also came up with a star-structure to organize these columns into different views using Snowflake. I have come up with 4 views: INVESTORS, COMPANIES, TIMES_TO_UNICORN, STATUS_CHANGES

![Schema](https://github.com/aliciango/data-analyst-portfolio-projects/blob/fcb5087ebe4dd2112108b45e5b47794e9a86ef64/assets/star%20schema.png)

By organizing data this way, it would be so much easier to query about a specific investor, company,  etc. I have included some queries that I have used to test the if the schema is working properly.

```sql
-- ==============================================================
-- QUERY TESTING
-- Comparing the growth of emerging markets: Brazil, Russia, India, China, and South Africa
-- with that of established markets
-- ==============================================================
WITH markets AS (
    SELECT 
        company_cleaned as company, 
        CASE 
            WHEN country IN ('Brazil', 'Russia', 'India', 'China', 'South Africa') 
            THEN 'emerging market' 
            ELSE 'established market' 
        END AS market_cat
    FROM view_companies
) 
SELECT  
    m.market_cat,
    -- average percentage change per industry
    ROUND(AVG(cf.pct_22_23), 2) AS avg_industry_pct_22_23,
    ROUND(AVG(cf.pct_23_24), 2) AS avg_industry_pct_23_24
FROM markets m
INNER JOIN view_companies_full cf 
    ON m.company = cf.company
GROUP BY m.market_cat
ORDER BY avg_industry_pct_22_23 DESC;
```

```sql
-- ==========================
-- the total valuation of companies that have data in all three years
-- TOTAL PORTFOLIO VALUATION 
-- ==============================================================
CREATE OR REPLACE VIEW total_portfol_vals AS (
    WITH valid_comp_investors AS (
        SELECT 
            c22.select_investors AS investor
            , c22.company_cleaned AS company
        FROM companies_ls_22 c22
            INNER JOIN companies_ls_23 c23 ON c22.company_cleaned = c23.company_cleaned AND c22.select_investors = c23.select_investors
            INNER JOIN companies_ls_24 c24 ON c24.company_cleaned = c22.company_cleaned AND c23.select_investors = c24.select_investors)
        
    SELECT 
        v.investor
        , SUM(cf.val_2022) total_portfol_val_22
        , SUM(cf.val_2023) total_portfol_val_23
        , SUM(cf.val_2024) total_portfol_val_24
    FROM valid_comp_investors v
    INNER JOIN view_companies_full cf 
        ON v.company = cf.company
    GROUP BY v.investor
    ORDER BY total_portfol_val_22 DESC
);

-- TOTAL PORTFOLIO VALUATION % CHANGE YoY
SELECT 
    *
    , ROUND( (total_portfol_val_23 - total_portfol_val_22)*100::numeric / total_portfol_val_22,2) as pct_portfol_val_22_23
    , ROUND( (total_portfol_val_24 - total_portfol_val_23)*100::numeric / total_portfol_val_23,2) as pct_portfol_val_22_23,
FROM total_portfol_vals;


```
However, designing the project this way cannot allow fast querying, joining for analysis. The goal is to serve the interests of investors by allowing them to view the snapshot of their investment portfolio and picking the company that will increase their value. Therefore, I created an additional view where I aggregated everything and called in 'One Big Table'


## 4. Tableau - Dashboard: Portfolio Management and Deal Sourcing
Using the one big table, I was able to create this visualization: [Tableau Dashboard](https://public.tableau.com/views/FINISHEDDASHBOARD/DEALSOURCINGDB?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


