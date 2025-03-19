import pandas as pd

new_query = pd.read_csv('../resources/new_training_questions_new_50.csv')
ddls = [
    """
    CREATE TABLE aggregated_data (
        CompanyCode int,
        Profit_Center int,
        Functional_Area int,
        YearPeriod int,
        Amount float,
        CompanyCodeName nvarchar(-1),
        FY int,
        Period nvarchar(-1),
        Cost_Center int,
        Func_Area_Desc nvarchar(-1),
        GL_Account_Name nvarchar(-1),
        GL_CostCenters_FuncArea nvarchar(-1),
        GL_FA nvarchar(-1),
        Segment nvarchar(-1),
        Profit_Center_Desc nvarchar(-1),
        ID int,
        GL_account int,
        Cost_Center_groups nvarchar(-1),
        Cost_Type nvarchar(-1),
        Join_Key nvarchar(-1),
        Excluded_pp int
    );
    """,
    """
       CREATE TABLE hierarchy (
            ID int,
            Level int,
            Parent_Id float,
            Level1 nvarchar(-1),
            Path nvarchar(-1),
            HierarchyDepth int,
            Has_child bit,
            Level2 nvarchar(-1),
            Level3 nvarchar(-1),
            Level4 nvarchar(-1),
            Level_1_2_3_4_5 nvarchar(-1),
            Level5 nvarchar(-1),
            IsOverhead bit,
            Overheads_Consolidation_Column nvarchar(-1),
            Level6 nvarchar(-1),
            item nvarchar(-1),
            hierarchy nvarchar(-1),
            r_flag float,
            Level_str money,
            Variable_costs_Test_filter bit);
    """,
    """
        CREATE TABLE calculated_table (
            ID int,
            Level int,
            Parent_Id float,
            Level1 nvarchar(-1),
            Path nvarchar(-1),
            HierarchyDepth int,
            Has_child bit,
            Level2 nvarchar(-1),
            Level3 nvarchar(-1),
            Level4 nvarchar(-1),
            Level_1_2_3_4_5 nvarchar(-1),
            Level5 nvarchar(-1),
            IsOverhead bit,
            Overheads_Consolidation_Column nvarchar(-1),
            Level6 nvarchar(-1),
            item nvarchar(-1),
            hierarchy nvarchar(-1),
            r_flag float,
            Level_str money,
            Variable_costs_Test_filter bit
        );
    """
]

aggregated_doc = """This dataset enables detailed monitoring and analysis of financial transactions across various profit centers and functional areas within a company. With a focus on attributes like CompanyCode, Profit_Center, Functional_Area, and Amount, it facilitates comprehensive financial reporting and performance evaluation over different fiscal periods. The structured data aids in combining and analyzing general ledger accounts, cost centers, and functional areas, making it easier to identify financial trends, perform comparative assessments, and support strategic decision-making in financial management.

The following are examples of the unique values found in the dataset:

Cost_Center_Desc:

Cost_Center_1
Cost_Center_10
Cost_Center_100
Cost_Center_1000
Cost_Center_1001
Cost_Center_1002
Cost_Center_1003
Cost_Center_1004
Cost_Center_1005
Cost_Center_1006
Cost_Center_1007
Cost_Center_1008
Cost_Center_1009
Cost_Center_101
Cost_Center_1010
Cost_Center_1011
Cost_Center_1012
Cost_Center_1013
Cost_Center_1014
Cost_Center_1015
Cost_Center_1016
Cost_Center_123

Func_Area_Desc:

Functional_Area_1
Functional_Area_10
Functional_Area_11
Functional_Area_12
Functional_Area_13
Functional_Area_14
Functional_Area_15
Functional_Area_16
Functional_Area_17
Functional_Area_18
Functional_Area_19
Functional_Area_2
Functional_Area_20
Functional_Area_21
Functional_Area_22
Functional_Area_23

Profit_Center_Desc:

Profit_Center_1
Profit_Center_10
Profit_Center_11
Profit_Center_12
Profit_Center_13
Profit_Center_14
Profit_Center_15
Profit_Center_16
Profit_Center_17
Profit_Center_18
Profit_Center_19
Profit_Center_2
Profit_Center_3
Profit_Center_4
Profit_Center_5
Profit_Center_6
Profit_Center_7
Profit_Center_8
Profit_Center_9
These examples represent a subset of the diverse and unique values present in the dataset."""
hierarchy_doc = """Hierarchy Table Description
The Hierarchy table contains detailed levels of cost structures within the organization, allowing for comprehensive financial analysis and reporting. The levels include:

Level1: Broad categories of costs

Non-controllable Costs
Non-Operating Cost
General & Admin Overheads
Controllable Contribution
Franchise & Sales Costs
Conversion Costs
Trade
Total GSV Third Party
R Depreciation
Provision for Tax
Advertising & Consumer Promo
Prime Costs
Affiliate Sales
Other Costs
Level2: Subcategories under each Level1

Non-controllable Costs Royalties
Non-controllable Costs X-charges
Non-controllable Costs JE Offsets
Process Technology Royalty Expense
Other Non-Operating Costs
Info Systems Overheads
S&F Overheads
Commercial Overheads
Corporate Affairs Overheads
Franchise Overheads
Sales Overheads
Logistics
External Manufacturing
Internal Manufacturing
Trade Expenditure
3rd Party Export GSV
Gross Sales 3rd Party Dom
Display & Equipment Costs
Consumer Promotions
Advertising
Carbon Offset COGS Prime
Co-Mfg Raws/Packs
Raws/Packs & Bought In
Affiliate Sales - Services
Affiliates Sales - Products
Donated/Destroyed Products
Total Benefit Plan Non-Op
Interest Net
Other Operating Costs
Legal Overheads
R&D Overheads
Other Depreciation
Engineering Overheads
P&O Overheads Total
Info Sys Hrdwr Depr X-charge
Intercompany Purchase Elimination
Level3: Further detailed subcategories under Level2

Royalties Expense
Royalties Income
Non Op Affiliate Non-NCFO
Process Technology Royalty Income
Formula Royalty Expense
Trademark Royalty Expense
MGS X-charges Expense - G&A
CSF X-Charges Expense
CSS X-Charges Expense - G&A
SSF X-Charges Expense
ICTC X-charges Expense - G&A
CSS X-charges Expense - Fr&Sales
Segment X-charges Expense - G&A
Segment X-charges Expense - Fr&Sales
Segment X-charges Expense- Advertising
Segment X-charges Expense - Conversion
Segment X-charges Income - G&A
Segment X-charges Income - Fr&Sales
Segment X-charges Income - Advertising
Segment X-charges Income - Conversion
MGS X-charges Income - Depr
MGS X-charges Income - G&A
MGS X-charges Income - Fr&Sales
ICTC X-charges Income - G&A
CSF X-Charges Income
MGS Offset Depreciation
MGS Offset F&S
Journal Entry CSS Offset
Journal Entry MGS offset
Other Non-Op Costs Non-NCFO
Non Op Costs – Project X-charge in/out
Non Op Costs – Project 81
Non Op Costs – Project 79
Non Op Costs – Project 78
Non Op Costs – Project 77
Non Op Costs – Project 76
Non Op Costs – Project 75
Non Op Costs – Project 74
Non Op Costs – Project 73
Non Op Costs – Project 71
Non Op Costs – Project 70
Non Op Costs – Project 69
Non Op Costs – Project 68
Non Op Costs – Project 67
Non Op Costs – Project 66
Non Op Costs – Project 64
Non Op Costs – Project 62
Non Op Costs – Project 61
Non Op Costs – Project 60
Non Op Costs – Project 59
Non Op Costs – Project 56
Non Op Costs – Project 54
Non Op Costs – Project 53
Non Op Costs – Project 52
Non Op Costs – Project 51
Non Op Costs – Project 50
Non Op Costs – Project 49
Non Op Costs – Project 45
Non Op Costs – Project 43
Non Op Costs – Project 40
Non Op Costs – Project 39
Non Op Costs – Project 36
Non Op Costs – Project 34
Non Op Costs – Project 33
Non Op Costs – Project 27
Non Op Costs – Project 25
Non Op Costs – Project 24
Non Op Costs – Project 18
Non Op Costs – Project 9
Non Op Costs – Project 7
Non Op Costs – Project 1
Global R&D True Up
Non-Operating MGS Cross Charge
Other Non-Op Inc/Exp. - 3rd
Interest Expense 3rd Party Total
Interest Income 3rd Party
Other Op All Other
Other Op IO Settlements
Other Op IC Charges
Other Op Costs
Level4: Additional breakdowns under Level3

Non Op Affiliate Expense Non-NCFO
Non Op Costs Exp Aff - Corp Project
Non Op Costs Inc Aff - Corp Project
Acquisition/Integration - Expense
Mgt Chg & Service Fees WWY Affil
Divestiture Expenses
Minority Interest(Inc)/Expense
Amort Int Dev Cap Soft Info
Office Services Project Expense
Office Services - Expenses
Office Services - SWB
Finance - Expenses
Finance - SWB
Market Head Expenses
Market Head SWB
Sales Support Expenses
Sales Support SWB
Logistics Ret Undep M&E - Hist
Logistics Hist Deprec Exp. M&E
Logistics Hist Deprec Exp. Bldg & Imp
Import Freight
Trunking ICB
Trunking ICB recovery
Trunking - Domestic
Logistics SWB
Logistics Expenses
Other Storage & Handling
In-house Warehouse Costs/Exp.
Handling
Storage
Logistic Labor
Co-Mfg./Co-Pckg Inven.Reval
Co-Mfg./Co-Pckg Non Quality
CoMfg/CoPkg Conv Service
Other Manufacturing Costs
Recall Discounts
Closeout
Markdown
Non-saleable Returns(Sale Val)
Coupons & Redemptions Discount
Shopper Price Promotion Discounts
Promotion Discounts
Sales Value Free Product in Trade
Growth & Development Incentives
Displays & Trade Advertising Discounts
Assortment and Placement Discounts
Listing Fees
Cash and Prompt Payment Discounts
Customer Logistic Discounts
Mktg Proceeds Sale Vend Equip
Mktg Retire Undep M & E Hist
Mktg Retire Undep Bldg & Imp Hist
Mktg Hist Deprec Exp Info Hrdwr
Mktg Hist Deprec Exp Furn & Fix
Mktg Hist Deprec Exp M & E
Mktg Hist Deprec Exp Bldg & Imp
Mktg Hist Deprec Vend Equip
Mktg Hist Deprec Exp Land & Imp
Sponsorship
Research and External Relations
Production
Agency Fees
Paid Media - Retailer & Search
Paid Media - Digital
Paid Media - Traditional
Damages & Discrepancies
Cost of Pallets
Administration (Log)
Logistic Expenses Write-off
Inventory Val. Logistics
Co-Mfg - Usage/Excess
PPV Co-Mfg. Raws/Packs
Std Co-Mfg. Raws/Packs
Inventory Valuation Prime
3rd Pty Purchases - FG
Intercompany Purchases
Packaging Costs
Fct Raws
Aff Sales Serv Info Services Ovhds
Affiliate Sales Services Franchise
Affiliate Sales Services Finance
Affiliate Sales Services P&O
Affiliate Sales Services Commercial
Aff Sales Serv Info Service Depr
Affiliate Sales Services R&D
Aff Sales - Prod Currency G/L
Aff Sales - Prod Currency Hedge
Aff Sales - Products excl Currency
Account Management Overheads
Sales SWB
Sales Expenses
Other Non-Op Costs NCFO
Ben Plan (Inc)/Exp Below Op
IARP Expense Below Operating
Mirror Prom Expense Below Op
Interest Inc/Exp. Affiliate
Amort of Intangibles
Engineering Capitalization
P&O Ovhds Services Support
P&O Ovhds Services Expense
P&O Ovhds Services SWB
P&O Services Project Expense
Non Op Costs
Total MGS Other Expense
Information Systems
Other Operating (Advertising)
Product Integrity Total
Level5: The most detailed level, providing specific expense items

Non Op Affiliate Income Non-NCFO
Trunking-Dom-IntDep Transf
Trunking - Dom. - Shuttles
Direct Labor Log - Contr/Temp
Manufacturing Depreciation
Prcd Sale Computer Equipment
Amort Int Dev Cap Soft Info
Direct Labour - Assoc.
Packs - Usage/Excess
Commercial Expenses
Service Desk/Commercial
Info Sys/Tech Expense
Procurement
Operating Supplies
Other General Factory Expenses
Raws and Packs Costs
Material Usage/Variance
Variable Manufacturing Overheads
Depreciation/Amort. of Assets
Utilities and Energy
Labor & Other Related Costs
Machine Related Costs
Quality Control Costs
Insurance Costs
Consumables
Maintenance
Environmental Costs
Packaging
Warehouse
Transportation
Engineering Costs
Operating Labor
Packaging Labor
Plant Labor
Repacking
Reprocessing
Warehouse Labor
Engineering Labor
Customer Specific Costs
Non-Operating MGS Cross Charge
Global R&D True Up
Non-Operating Info System
Non-Operating Other
Total Non-Operating Costs
This hierarchical structure allows for detailed monitoring and analysis of expenses and sales performance over time, facilitating comparative assessments within and across fiscal years, and aiding in identifying trends and anomalies for strategic decision-making in financial management and sales optimization."""

relation_doc = """NSV Third Party = 	GSV Third Party - Trade
 
Total NSV = Affiliate Sales + GSV Third Party - Trade
 
Prime Margin = Affiliate Sales + GSV Third Party - Trade-Prime Cost
 
Margin After Conversion =Affiliate Sales + GSV Third Party - Trade-Prime Cost- Conversion Costs
Controllable Contribution	Margin after Contribution - Adverstising & consumer promo - Franchise and sales cost
 
Controllable Profit = Margin after Contribution - Adverstising & consumer promo - Franchise and sales cost- General and administrative overheads - Other (Income) Costs
 
Controllable Earnings = Margin after Contribution - Adverstising & consumer promo - Franchise and sales cost- General and administrative overheads - Other (Income) Costs - Depreciation
 
Profit Before Tax (PBT) = Margin after Contribution - Adverstising & consumer promo - Franchise and sales cost- General and administrative overheads - Other (Income) Costs - Depreciation
 - Non-operating cost - non controllable costs
 
 
 
Net Income (PAT) =	Margin after Contribution - Adverstising & consumer promo - Franchise and sales cost- General and administrative overheads - Other (Income) Costs - Depreciation
 - Non-operating cost - non controllable costs- Provision for tax"""

naming_conventions = """ Accounting profile is Fucn_Area_Desc, If any question contains "Accounting profile," it is referring to Func_Area_Desc"""

connection_doc = """To ensure accurate query interpretation, if the user provides a string value such as Profit_Center_9, Functional_Area_23, or Cost_Center_123, the model should reference the corresponding description columns: Profit_Center_Desc, Func_Area_Desc, and Cost_Center_Desc"""

yearperiod_doc = "Always format the YearPeriod as YYYY00P, where YYYY is the year and P is the period, when writing SQL queries. For example,  2022 period 1 should be 2022001 and 2022 period 13 should be 2022013."