[Go to Tableau Dashboard](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/viz_hr/FINALDB?publish=yes)

<img src="https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/hr-dashboard.png" alt="img alt" width="600" height="400">

# Brazillian Olist Ecommerce Project
### Using Ecommerce Data for Analyzing Buyer Retention and Seller Acquisition
- Report to Business Stakeholders: [Go to Business Insights](#bulb-business-insights)
- Report to Tech Stakeholders: [Go to Data Processing](#toolbox-data-processing)


# :bulb: BUSINESS INSIGHTS
[Scroll down for business recommendations](#bulb-recommendations)
## Dashboard Peek
:link: [Go to Tableau Dashboard](https://public.tableau.com/views/MarketplaceEcommerceAnalysis_SellerAcquisitionCustomerRetention/SELLERBUYERNETWORKDB?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


## Northstar Metrics: Monthly Income

## Insights from Visualization

1. Primary Driver: Compensation
- The most critical insight comes from the top-right chart ("Monthly Income is the Reason..."):

- Inverse Relationship: There is a drastic inverse correlation between salary and attrition.

- The Danger Zone: The vast majority of employees leaving (137 people) fall into the lowest income bracket ($1k–$4k).

- The Drop-off: As soon as salary exceeds $7k, attrition drops to near zero. This suggests that once employees cross a specific financial threshold, retention stabilizes significantly.

2. Demographic Profile of Attrition
- The employees leaving are not just "low earners"; they fit a specific demographic profile often associated with entry-level positions:

- Age: The 21–30 and Under 20 age brackets show a visibly higher proportion of attrition compared to older groups (31+).

- Experience: Employees with 0–4 years of working experience see high turnover.

- Education: Turnover is spread relatively evenly across education levels, suggesting that higher education does not necessarily insulate an employee from leaving if the pay is low.

3. Structural/Career Progression Issues
- The bottom-right charts ("Defining factors for High Income") reveal why the lower-income group might be frustrated:

- Seniority-Based Pay: There is a strong positive correlation between Total Working Years and Monthly Income.

- Interpretation: If income is strictly tied to tenure (time served) rather than performance, high-performing young talent in the $1k–$4k bracket may feel they cannot advance their salary fast enough, prompting them to leave for better offers elsewhere.

4. Departmental Hotspots
- Job Roles: Sales Executives, Research Scientists, and Laboratory Technicians have the highest volume of attrition.

- Department: The R&D department has the highest volume of turnover, though this is likely because they also have the highest headcount.





## :bulb: Recommendations
- Review Entry-Level Compensation: Immediate market adjustment is needed for the $1k–$4k salary band. This is where the company is "bleeding" talent.
- Fast-Track Progression: Since income is currently tied heavily to tenure, consider implementing performance-based bonuses or faster promotion tracks for high-performing young employees to prevent them from leaving before they reach the $7k "safe zone."
- Focus on Retention in Sales & Lab Techs: Conduct exit interviews specifically for Sales Executives and Lab Technicians to see if non-monetary factors (burnout, shift work) are compounding the salary issue.


# :toolbox: DATA PROCESSING
## Tech Stack
- Python
- [Tableau (Visualization)](https://public.tableau.com/app/profile/alicia.ngo.viz/viz/MarketplaceEcommerceAnalysis_SellerAcquisitionCustomerRetention/HOMEDB)

## 1. Data Source Introduction

### 1a. Data Source Link
The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

- [IBM Human Resource Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Brainstorming on how to extract insights from those columns using [Go to Excel](https://docs.google.com/spreadsheets/d/1_EEFbJIkdSVwN5KGzisHomI69wjRjiG07iNR6OZ8RqY/edit?usp=sharing)

## 2. Python - Data Cleaning
- I have taken reference of this link to conduct dimension reduction: [GeekForGeek](https://www.geeksforgeeks.org/machine-learning/ibm-hr-analytics-on-employee-attrition-performance-using-random-forest-classifier/)
- First, I used random forest to identify the most significant predictor of attrition, which is 'Monthly Income'. This means that the higher the income, the more motivated an employee is to stay with IBM.
- Next, I dived deeper into understanding drivers behind high income. A high income is determined by the job level and surprisingly the total working years and salary does not have perfectly linear relationship. This means that regardless of the years that employee contributes to IBM, it is not guaranteed that they will get the pay raise. This suggests for a better policy to retain and reward employees who have stayed with IBM for a long time.

![img alt](https://github.com/trucngo100/data-analyst-portfolio-projects/blob/main/assets/hr-income_regardless.png)
## Technical Challenges
- There are more than 20 categorical variables in this dataset and to pin down the variables that are highly correlated with the attrition was the main challenge.
- First, I conducted dimension reduction by applying random forest. From there

- Personal thought: the most exciting part while doing this project is that I got to learn so much about using AI to write difficult Machine Learning code. Still, interpreting and coming up with the right questions to ask was the essence of the great output. The AI-generated code was not correct, so I have to go back and forth to check the logic and make any tailored adjustments.
- There were some variables that I have not touched on yet, and they are the variables that reflect employee's satisfaction about their job.


