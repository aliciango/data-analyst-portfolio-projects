# Unicorn Companies Project
- Report to Business Stakeholders: [Go to Business Insights](#bulb-business-insights)
- Report to Tech Stakeholders: [Go to Data Processing](#toolbox-data-processing)


# :bulb: BUSINESS INSIGHTS
[Scroll down for Business Recommendations](#bulb-recommendations)

## Dashboard Peek
:link: [Go to Tableau Dashboard](https://public.tableau.com/views/FINISHEDDASHBOARD/DEALSOURCINGDB?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

| ![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/d936e06e34322df8660598fec4c685f3698567cf/assets/unicorn_tableau.png) |
|------------------------|

## Northstar Metrics: Company's Valuation in 2022, 2023 and 2024

- Dimension 1: Slice by Industries
- Dimension 2: Slice by Continents


## Insights from Visualization
#### *VIEW 1* - Portfolio Management Overview: Key Performance Indicators
| ![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/18a0f918f9d4dbbed7a932b820d52477a9639745/assets/sequoia%20overview.png)  |
|------------------------|
##### Executive Summary (KPIs)
- Portfolio Health: The portfolio is highly concentrated, with only 31 holdings and the majority of holdings are in tech industry. This concentration hints that any changes in our portfolio may be caused by the recent changes in the tech industry. There was a major drawdown in 2023, with the portfolio's value dropping ~26% (from $417B to $309B), likely reflecting a broad market correction in tech.
- Rebound: The portfolio saw a 10.7% recovery in 2024 (from $309B to $342B). While positive, this has not yet erased the 2023 losses, and the valuation did not reach its peak as in 2022.
  
##### Funding Allocations to different Industry (Pie Chart)
- The portfolio has a massive sector concentration in Fintech (30.65%). The top three sectors (Fintech, Supply Chain, and Internet Software) collectively account for over 55% of the portfolio.
- This heavy concentration in Fintech poses certain risks to the portfolio. A downturn in this specific sector will severely impact overall performance. Conversely, it also represents a strong bullish bet.

##### Industry YoY Growth from 2022 to 2024 (Slope Chart)
- This chart gives explanation for whhy the total portfolio valuation dropped in 2023. Nearly every single key industry in the portfolio (Fintech, Internet Software, E-commerce, Supply Chain) experienced a sharp decline in value from 2022 to 2023.
- These 4 industry's trends also seem to move in tandem and it reflects the interconnectivity and interdependency among these industries.

##### Funding Allocations to different Continents (Bubble Chart and Map)
- There is a strong geographic concentration in North America (59.55%), with a significant secondary allocation to Europe (23.03%).
- This exposes the portfolio primarily to North American and European economic conditions and political situations. Asia, South America, and Africa are significantly under-allocated.

##### Continent YoY Growth from 2022 to 2024 (Slope Chart)
- The 2023 downturn was global, as valuations in North America, Asia, and Europe all declined. However, North America shows the strongest rebound in 2024, recovering more steeply than Asia or Europe.
- This chart suggests the heavy allocation to North America (from the previous chart) is currently driving the portfolio's recovery from the downturn in 2023 to 2024.

#### *VIEW 2* - Portfolio Management: Industry Snapshot
| ![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/18a0f918f9d4dbbed7a932b820d52477a9639745/assets/sequoia%20industry.png) |
|------------------------|
##### Chart Industry's Years to Unicorn
This allows investors with different investment taste to understand which industries suit their taste.
- High risk, high return, short holding: An investor seeking a faster potential return (acknowledging higher risk) should look at sectors at the top of the list, like Hardware (4.9 years) or Consumer & Retail (5.4 years).
- Low risk, low return, long-term holding: An investor with a more patient, long-term thesis might be comfortable with sectors like Travel (8.2 years) or Internet Software (8.0 years).

##### Chart Company Valuation versus Industry Average Valuation
This chart shows the breakdown of the portfolio. The portfolio's health is driven by massive outliers at both ends.
- Hyper-Winners: The portfolio contains enormous winners that significantly skew its value. The prime example is Klarna (Fintech), valued at $46B in 2022 compared to its industry average of $4.7B. Similarly, Wiz (Cybersecurity) and Attentive (Mobile & Tele...) are valued at 2x their respective industry averages.
- Underperformers: The portfolio also holds companies valued below their peers, such as Snapdocs and Remote (both in Fintech), which are underperforming the average.

##### Chart Funding Allocations to different Industries (% Change Waterfall)
- The portfolio's -26% drop from 2023 to 2024 was driven almost entirely by the -22.5% collapse in Fintech and the -14.3% drop in Supply Chain. Given that Fintech is a massive holding (as seen in the first dashboard and confirmed by Klarna's size), this downturn was the primary cause of the portfolio's 2023 losses.
- The 2023 Savior: The portfolio was prevented from an even worse fate by the explosive +66.7% growth in Cybersecurity (likely driven by the holding in Wiz). This sector's stellar performance acted as a partial hedge.
- The 2024 Recovery: The recovery is broad-based. Fintech, Supply Chain, and Hardware are all rebounding sharply (+16.7%), signaling a "risk-on" sentiment returning to these sectors.
- Other than that, some industries can be potential candidates because they show significant growth over the last two years, such as Cybersecurity growing +66.7% and then another +4.5%.

#### *VIEW 3* - Portfolio Management: Continent Snapshot
| ![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/18a0f918f9d4dbbed7a932b820d52477a9639745/assets/sequoia%20continent.png) |
|------------------------|
##### Chart Years to Unicorn
- This benchmark chart shows the average time it takes for a startup to reach a $1B valuation in various countries.
- High-velocity opportunities exist in both established markets (Philippines at 2.0 years) and emerging markets (UAE and Nigeria at 3.0 years).

##### Chart Company Valuation versus Continent Average Valuation
- Bytedance ($180B) and Shein ($100B) are valued at multiples (24x and 20x, respectively) of their peer averages.
- This means the portfolio's entire "Asia" valuation is almost completely dependent on the private valuations of just two companies. This is a significant concentration risk.
- In contrast, the Africa holding (Opay at $2B) is valued at less than half its industry average ($4.7B), indicating a severe underperformer.

##### Chart Funding Allocations to Different Industries
- The portfolio's 2024 value was driven by Asia, which grew an explosive +35.6%. This is likely due to valuation mark-ups in Bytedance and Shein.
- Other regions like North America (+13.7%) and Europe (+12.7%) are also posting solid, healthy recoveries.
- Africa is the only region that is still declining, posting a -21.4% loss in 2024. This confirms that the weakness seen in Opay is dragging down the entire continent's allocation.

#### *VIEW 4* - Portfolio Management: Deal Sourcing
| ![image alt](https://github.com/aliciango/data-analyst-portfolio-projects/blob/18a0f918f9d4dbbed7a932b820d52477a9639745/assets/sequoia%20deal%20sourcing.png)|
|------------------------|
Investors can view the in-portfolio companies and out-portfolio companies. They can also inspect each company by seeing the details in the bottom lookup table.



## :bulb: Recommendations

Based on the insights, here are the proposed strategic investment opportunities:

- As the portfolio is highly allocated to the tech industry, investors can consider diversification as a strategy to improve their portfolio health by adding a company in an under-allocated sector. Opportunities for diversification exist in Artificial Intelligence (only 5.76%) or other unlisted, non-correlated industries (represented by "other" at 0.81%)
- Find the next Wiz or Klarna to pick as a portfolio company. In the meantime, investors should also re-evaluate underperformers like Snapdocs.
- Use continent as a factor to consider when conducting deal sourcing: If the investors prefer fast return, consider adding to their portfolios companies located in the Philippines, UAE, or Nigeria. On the other hand, Korean and Brazilian companies can be good options if investors are seeking for low but sustainable returns.
- The Africa allocation is failing. An investor could make a contrarian bet by finding a high-potential African startup (perhaps in Nigeria, from Chart 1) to turn this performance around. Alternatively, the portfolio has very little exposure to South America (+8.6%), which could be a valuable, long-term diversification play.


# :toolbox: DATA PROCESSING

## Tech Stack
- Python (pandas and numpy for data cleaning)
- [Snowflake/SQL](#snowflake) (database design, merge tables, quick querying)
- [Tableau (Visualization)](https://public.tableau.com/views/FINISHEDDASHBOARD/DEALSOURCINGDB?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 1. Data Source Introduction

### 1a. Data Source Link
Each dataset contains over 1000+ unicorn companies with valuation over $1 and I have merged three different datasets that were crawled in 2022, 2023 and 2024.
- [2024 unicorn data](https://www.kaggle.com/datasets/sashakorovkina/2024-unicorn-and-emerging-unicorn-companies?resource=download&select=unicorn_companies.csv)
- [2023 unicorn data](https://www.kaggle.com/datasets/ritwikb3/unicorn-companies)
- [2022 unicorn data](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies)

3 CSV files that collect information about private companies with a value of over $1 billion as of March 2022, 2023, and 2024.

### 1b. Variable Description
I used Excel to organize the data schema and brainstorm how can I utilize this dataset.
[Go to Excel](https://docs.google.com/spreadsheets/d/1EC9eJmgNzuyrkrtsYNhkAlcCXkZZVeCBpEFL-eOE2og/edit?usp=sharing)
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


