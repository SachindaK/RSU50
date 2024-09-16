import pandas as pd

# Load Excel files
textile_exports_file = '/content/drive/MyDrive/RSU50 DataFiles/Textile_Exports.xlsx'
world_indicators_file = '/content/drive/MyDrive/RSU50 DataFiles/World_Development_Indicators.xlsx'

# Read Excel files
textile_exports_df = pd.read_excel(textile_exports_file)
world_indicators_df = pd.read_excel(world_indicators_file)

# Rename columns for consistency
textile_exports_df.columns = ['Country', 'Year', 'Textile Exports (US$ Thousand)']
world_indicators_df.columns = ['Country', 'Year'] + list(world_indicators_df.columns[2:])

# Merge datasets on 'Country' and 'Year'
merged_df = pd.merge(textile_exports_df, world_indicators_df, on=['Country', 'Year'])

# Remove columns with more than 40% missing values
threshold = len(merged_df) * .6
merged_df_clean = merged_df.dropna(thresh=threshold, axis=1)

# Select only numeric columns for correlation
numeric_columns = merged_df_clean.select_dtypes(include=['float64', 'int64']).columns

# Calculate correlations between textile exports and other numeric columns
correlation_results = merged_df_clean[numeric_columns].corr()['Textile Exports (US$ Thousand)'].drop('Textile Exports (US$ Thousand)')

# Find the 20 highest positive correlations and 20 lowest negative correlations
highest_positive_correlations = correlation_results.nlargest(20)
lowest_negative_correlations = correlation_results.nsmallest(20)

# Display the highest positive and lowest negative correlations
print("20 Highest Positive Correlations:")
print(highest_positive_correlations)

print("\n20 Lowest Negative Correlations:")
print(lowest_negative_correlations)

# 20 Highest Positive Correlations:
# CO2 emissions (kt) [EN.ATM.CO2E.KT]                                                                         0.826430
# Total greenhouse gas emissions (kt of CO2 equivalent) [EN.ATM.GHGT.KT.CE]                                   0.820694
# Methane emissions in energy sector (thousand metric tons of CO2 equivalent) [EN.ATM.METH.EG.KT.CE]          0.812795
# Nitrous oxide emissions in energy sector (thousand metric tons of CO2 equivalent) [EN.ATM.NOXE.EG.KT.CE]    0.806136
# Merchandise imports (current US$) [TM.VAL.MRCH.CD.WT]                                                       0.792812
# International tourism, number of arrivals [ST.INT.ARVL]                                                     0.791348
# Merchandise imports by the reporting economy (current US$) [TM.VAL.MRCH.WL.CD]                              0.789831
# Scientific and technical journal articles [IP.JRN.ARTC.SC]                                                  0.789022
# Adjusted net savings, including particulate emission damage (current US$) [NY.ADJ.SVNG.CD]                  0.787269
# Adjusted net savings, excluding particulate emission damage (current US$) [NY.ADJ.SVNX.CD]                  0.786135
# Goods imports (BoP, current US$) [BM.GSR.MRCH.CD]                                                           0.781789
# Adjusted net national income (current US$) [NY.ADJ.NNTY.CD]                                                 0.777234
# Exports of goods, services and primary income (BoP, current US$) [BX.GSR.TOTL.CD]                           0.774689
# Imports of goods, services and primary income (BoP, current US$) [BM.GSR.TOTL.CD]                           0.773586
# Exports of goods and services (BoP, current US$) [BX.GSR.GNFS.CD]                                           0.769017
# Imports of goods and services (BoP, current US$) [BM.GSR.GNFS.CD]                                           0.768917
# Merchandise exports (current US$) [TX.VAL.MRCH.CD.WT]                                                       0.767951
# Goods exports (BoP, current US$) [BX.GSR.MRCH.CD]                                                           0.766715
# Foreign direct investment, net inflows (BoP, current US$) [BX.KLT.DINV.CD.WD]                               0.764567
# Merchandise exports by the reporting economy (current US$) [TX.VAL.MRCH.WL.CD]                              0.762646
# Name: Textile Exports (US$ Thousand), dtype: float64

# 20 Lowest Negative Correlations:
# Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent) [EN.ATM.GHGO.KT.CE]   -0.701532
# Population ages 10-14, female (% of female population) [SP.POP.1014.FE.5Y]                                      -0.609629
# Population ages 15-19, female (% of female population) [SP.POP.1519.FE.5Y]                                      -0.604442
# Population ages 0-14, female (% of female population) [SP.POP.0014.FE.ZS]                                       -0.603035
# Foreign direct investment, net (BoP, current US$) [BN.KLT.DINV.CD]                                              -0.600416
# Population ages 05-09, female (% of female population) [SP.POP.0509.FE.5Y]                                      -0.596491
# Population ages 0-14 (% of total population) [SP.POP.0014.TO.ZS]                                                -0.588333
# Population ages 15-19, male (% of male population) [SP.POP.1519.MA.5Y]                                          -0.582860
# Population ages 10-14, male (% of male population) [SP.POP.1014.MA.5Y]                                          -0.574949
# Population ages 0-14, male (% of male population) [SP.POP.0014.MA.ZS]                                           -0.571967
# Population ages 05-09, male (% of male population) [SP.POP.0509.MA.5Y]                                          -0.564353
# Population ages 00-04, female (% of female population) [SP.POP.0004.FE.5Y]                                      -0.560590
# Age dependency ratio, young (% of working-age population) [SP.POP.DPND.YG]                                      -0.536880
# Population ages 00-04, male (% of male population) [SP.POP.0004.MA.5Y]                                          -0.531377
# Mortality rate, adult, male (per 1,000 male adults) [SP.DYN.AMRT.MA]                                            -0.512667
# Mortality rate, adult, female (per 1,000 female adults) [SP.DYN.AMRT.FE]                                        -0.508096
# Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]                                                           -0.507166
# Age dependency ratio (% of working-age population) [SP.POP.DPND]                                                -0.481481
# Prevalence of anemia among pregnant women (%) [SH.PRG.ANEM]                                                     -0.473010
# Population ages 20-24, female (% of female population) [SP.POP.2024.FE.5Y]                                      -0.454143
# Name: Textile Exports (US$ Thousand), dtype: float64

# Define keywords that are related to the economy
economic_keywords = ['GDP', 'inflation', 'investment', 'employment', 'unemployment', 'exports', 'imports', 'business'
                     'trade', 'income', 'expenditure', 'monetary', 'interest rate', 'balance of payments']

# Filter the correlation results to find indicators that match economic keywords
economic_indicators = correlation_results[correlation_results.index.str.contains('|'.join(economic_keywords), case=False)]

# Find the 20 highest positive correlations and 10 lowest negative correlations
highest_positive_correlations = economic_indicators.nlargest(20)
lowest_negative_correlations = economic_indicators.nsmallest(20)

# Display the highest positive and lowest negative correlations
print("20 Highest Positive Correlations:")
print(highest_positive_correlations)

print("\n20 Lowest Negative Correlations:")
print(lowest_negative_correlations)

# 20 Highest Positive Correlations:
# Merchandise imports (current US$) [TM.VAL.MRCH.CD.WT]                                0.792812
# Merchandise imports by the reporting economy (current US$) [TM.VAL.MRCH.WL.CD]       0.789831
# Goods imports (BoP, current US$) [BM.GSR.MRCH.CD]                                    0.781789
# Adjusted net national income (current US$) [NY.ADJ.NNTY.CD]                          0.777234
# Exports of goods, services and primary income (BoP, current US$) [BX.GSR.TOTL.CD]    0.774689
# Imports of goods, services and primary income (BoP, current US$) [BM.GSR.TOTL.CD]    0.773586
# Exports of goods and services (BoP, current US$) [BX.GSR.GNFS.CD]                    0.769017
# Imports of goods and services (BoP, current US$) [BM.GSR.GNFS.CD]                    0.768917
# Merchandise exports (current US$) [TX.VAL.MRCH.CD.WT]                                0.767951
# Goods exports (BoP, current US$) [BX.GSR.MRCH.CD]                                    0.766715
# Foreign direct investment, net inflows (BoP, current US$) [BX.KLT.DINV.CD.WD]        0.764567
# Merchandise exports by the reporting economy (current US$) [TX.VAL.MRCH.WL.CD]       0.762646
# Military expenditure (current USD) [MS.MIL.XPND.CD]                                  0.752124
# Imports of goods and services (current US$) [NE.IMP.GNFS.CD]                         0.752103
# Exports of goods and services (current US$) [NE.EXP.GNFS.CD]                         0.750667
# Broad money (% of GDP) [FM.LBL.BMNY.GD.ZS]                                           0.749947
# Primary income payments (BoP, current US$) [BM.GSR.FCTY.CD]                          0.746909
# Primary income receipts (BoP, current US$) [BX.GSR.FCTY.CD]                          0.734949
# Domestic credit to private sector (% of GDP) [FS.AST.PRVT.GD.ZS]                     0.685559
# Commercial service imports (current US$) [TM.VAL.SERV.CD.WT]                         0.682198
# Name: Textile Exports (US$ Thousand), dtype: float64

# 20 Lowest Negative Correlations:
# Foreign direct investment, net (BoP, current US$) [BN.KLT.DINV.CD]                                        -0.600416
# Lending interest rate (%) [FR.INR.LEND]                                                                   -0.446734
# Agriculture, forestry, and fishing, value added (% of GDP) [NV.AGR.TOTL.ZS]                               -0.409065
# Vulnerable employment, female (% of female employment) (modeled ILO estimate) [SL.EMP.VULN.FE.ZS]         -0.405227
# Self-employed, female (% of female employment) (modeled ILO estimate) [SL.EMP.SELF.FE.ZS]                 -0.398558
# Food imports (% of merchandise imports) [TM.VAL.FOOD.ZS.UN]                                               -0.391526
# Vulnerable employment, total (% of total employment) (modeled ILO estimate) [SL.EMP.VULN.ZS]              -0.377067
# Employment in agriculture (% of total employment) (modeled ILO estimate) [SL.AGR.EMPL.ZS]                 -0.376202
# Contributing family workers, female (% of female employment) (modeled ILO estimate) [SL.FAM.WORK.FE.ZS]   -0.375033
# Transport services (% of commercial service imports) [TM.VAL.TRAN.ZS.WT]                                  -0.370454
# Employment in agriculture, female (% of female employment) (modeled ILO estimate) [SL.AGR.EMPL.FE.ZS]     -0.370380
# Transport services (% of service imports, BoP) [BM.GSR.TRAN.ZS]                                           -0.369425
# Self-employed, total (% of total employment) (modeled ILO estimate) [SL.EMP.SELF.ZS]                      -0.367297
# Employment in agriculture, male (% of male employment) (modeled ILO estimate) [SL.AGR.EMPL.MA.ZS]         -0.362113
# Contributing family workers, total (% of total employment) (modeled ILO estimate) [SL.FAM.WORK.ZS]        -0.360752
# Vulnerable employment, male (% of male employment) (modeled ILO estimate) [SL.EMP.VULN.MA.ZS]             -0.360431
# Final consumption expenditure (% of GDP) [NE.CON.TOTL.ZS]                                                 -0.350536
# Contributing family workers, male (% of male employment) (modeled ILO estimate) [SL.FAM.WORK.MA.ZS]       -0.348731
# Self-employed, male (% of male employment) (modeled ILO estimate) [SL.EMP.SELF.MA.ZS]                     -0.342103
# External health expenditure (% of current health expenditure) [SH.XPD.EHEX.CH.ZS]                         -0.339915
# Name: Textile Exports (US$ Thousand), dtype: float64
