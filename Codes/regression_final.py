from sklearn.linear_model import LinearRegression
import csv
import os
import locale
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import tables
from linearmodels import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8') 

base_dir = '..//data//WithoutReviews//'
dealData = base_dir + 'monotonic//'
metaData = base_dir + 'monotonic_meta//'
diff_Time = 3600000
time_base = 1546300800 ## Jan 1, 2019 Midnight (in ms)
days = 86400000

x = []
xx = []

def x_values(file_names, timediff, dealData, metaData):
	
	count = 0
	y, c, z, fixed_effects = ([] for i in range(4))
	deal_id, asin, no_type, numrev, avgrat, dealdis, actualdis, timeRem, recordtime, day = ([] for i in range(10))
	# r1_rating, r1_hrPred, r1_help, r1_ratio, r1_tokStd, r1_tokMean, r1_numTokens, r1_numChar = ([] for i in range(8))
	# r2_rating, r2_hrPred, r2_help, r2_ratio, r2_tokStd, r2_tokMean, r2_numTokens, r2_numChar = ([] for i in range(8))
	# r3_rating, r3_hrPred, r3_help, r3_ratio, r3_tokStd, r3_tokMean, r3_numTokens, r3_numChar = ([] for i in range(8))
	# r4_rating, r4_hrPred, r4_help, r4_ratio, r4_tokStd, r4_tokMean, r4_numTokens, r4_numChar = ([] for i in range(8))
	# r5_rating, r5_hrPred, r5_help, r5_ratio, r5_tokStd, r5_tokMean, r5_numTokens, r5_numChar = ([] for i in range(8))
	
	for j in file_names:
	
		n = 0

		with open(metaData + j, 'r', encoding='utf-8') as myfile:

			rows = []
			csvreader = csv.reader(myfile)

			for row in csvreader:
				rows.append(row)

			rating = float(rows[20][1])
			number_rev = int(rows[21][1])
			num_type = int(rows[4][1])
			orgprice = float(rows[15][1])
			disprice = float(rows[13][1])
			dealprice = float(rows[14][1])
			actualdiscount = float(((orgprice - disprice)*100)/orgprice)
			dealdiscount = float(((orgprice - dealprice)*100)/orgprice)
			dealID = rows[0][1]
			ASIN = rows[2][1]
			weekday = int(((int(rows[19][1]) - int(time_base))/days))%7
			# r1_NumTokens = float(rows[23][0])
			# r1_NumChar = float(rows[23][1])
			# r1_TokMean = float(rows[23][2])
			# r1_TokStd = float(rows[23][3])
			# r1_Ratio = float(rows[23][4])
			# r1_Help = float(rows[23][6])
			# r1_HrPred = float(rows[23][8])
			# r1_Rating = float(rows[23][9])
			# r2_NumTokens = float(rows[24][0])
			# r2_NumChar = float(rows[24][1])
			# r2_TokMean = float(rows[24][2])
			# r2_TokStd = float(rows[24][3])
			# r2_Ratio = float(rows[24][4])
			# r2_Help = float(rows[24][6])
			# r2_HrPred = float(rows[24][8])
			# r2_Rating = float(rows[24][9])
			# r3_NumTokens = float(rows[25][0])
			# r3_NumChar = float(rows[25][1])
			# r3_TokMean = float(rows[25][2])
			# r3_TokStd = float(rows[25][3])
			# r3_Ratio = float(rows[25][4])
			# r3_Help = float(rows[25][6])
			# r3_HrPred = float(rows[25][8])
			# r3_Rating = float(rows[25][9])
			# r4_NumTokens = float(rows[26][0])
			# r4_NumChar = float(rows[26][1])
			# r4_TokMean = float(rows[26][2])
			# r4_TokStd = float(rows[26][3])
			# r4_Ratio = float(rows[26][4])
			# r4_Help = float(rows[26][6])
			# r4_HrPred = float(rows[26][8])
			# r4_Rating = float(rows[26][9])
			# r5_NumTokens = float(rows[27][0])
			# r5_NumChar = float(rows[27][1])
			# r5_TokMean = float(rows[27][2])
			# r5_TokStd = float(rows[27][3])
			# r5_Ratio = float(rows[27][4])
			# r5_Help = float(rows[27][6])
			# r5_HrPred = float(rows[27][8])
			# r5_Rating = float(rows[27][9])

		myfile.close()

		with open(dealData + j, 'r', encoding='utf-8') as myfile:

			prevclaim = 0
			rows = []
			csvreader = csv.reader(myfile)

			for row in csvreader:
				rows.append(row)

			rownum = len(rows)
			prevtime = int(rows[1][9])
			prevdeal = int(rows[1][8])
			# prev = prevtime

			for row in rows[2:rownum]:

				if(int(row[8]) == 100):
					break

				if((prevtime - int(row[9])) >= timediff):
					prevclaim = int(row[8]) - prevdeal
					
					if(prevclaim < 0):
						prevclaim = 0

					fixed_effects.append(count)
					day.append(weekday)
					daytime = (int(row[11]) - int(time_base))%days
					y.append(prevclaim)
					avgrat.append(rating)
					numrev.append(number_rev)
					no_type.append(num_type)
					actualdis.append(actualdiscount)
					dealdis.append(dealdiscount)
					c.append(prevdeal)
					timeRem.append(int(row[9]))
					deal_id.append(dealID)
					asin.append(ASIN)
					# r1_rating.append(r1_Rating)
					# r1_hrPred.append(r1_HrPred)
					# r1_help.append(r1_Help)
					# r1_ratio.append(r1_Ratio)
					# r1_tokStd.append(r1_TokStd)
					# r1_tokMean.append(r1_TokMean)
					# r1_numTokens.append(r1_NumTokens)
					# r1_numChar.append(r1_NumChar)
					# r2_rating.append(r2_Rating)
					# r2_hrPred.append(r2_HrPred)
					# r2_help.append(r2_Help)
					# r2_ratio.append(r2_Ratio)
					# r2_tokStd.append(r2_TokStd)
					# r2_tokMean.append(r2_TokMean)
					# r2_numTokens.append(r2_NumTokens)
					# r2_numChar.append(r2_NumChar)
					# r3_rating.append(r3_Rating)
					# r3_hrPred.append(r3_HrPred)
					# r3_help.append(r3_Help)
					# r3_ratio.append(r3_Ratio)
					# r3_tokStd.append(r3_TokStd)
					# r3_tokMean.append(r3_TokMean)
					# r3_numTokens.append(r3_NumTokens)
					# r3_numChar.append(r3_NumChar)
					# r4_rating.append(r4_Rating)
					# r4_hrPred.append(r4_HrPred)
					# r4_help.append(r4_Help)
					# r4_ratio.append(r4_Ratio)
					# r4_tokStd.append(r4_TokStd)
					# r4_tokMean.append(r4_TokMean)
					# r4_numTokens.append(r4_NumTokens)
					# r4_numChar.append(r4_NumChar)
					# r5_rating.append(r5_Rating)
					# r5_hrPred.append(r5_HrPred)
					# r5_help.append(r5_Help)
					# r5_ratio.append(r5_Ratio)
					# r5_tokStd.append(r5_TokStd)
					# r5_tokMean.append(r5_TokMean)
					# r5_numTokens.append(r5_NumTokens)
					# r5_numChar.append(r5_NumChar)
					# val = (prevclaim, rating, number_rev, num_type, actualdiscount, dealdiscount, prevdeal, int(row[9]))

					if(daytime < 28800000):		# 12 am - 8 am
						recordtime.append(0)
					elif(daytime <= 43200000):		# 8 am - 12 pm (noon)
						recordtime.append(1)
					elif(daytime <= 57600000):		# 12 pm - 4 pm
						recordtime.append(2)
					elif(daytime <= 75600000):		# 4 pm - 9 pm
						recordtime.append(3)
					else:								# 9 pm - 12 am (midnight)
						recordtime.append(4)

					# for key, value in similar_val.items():
					# 	if(key == val):
					# 		print(key, similar_val[val], val, j)
					# 		break

					# similar_val[val] = j

					prevtime = int(row[9])
					prevdeal = int(row[8])

					# interaction.append(flag*(int(row[1])))
					# z.append(flag)
					# daytime = int(row[11])
			count = count + 1

	return [c, numrev, avgrat, actualdis, dealdis, timeRem, no_type, recordtime, fixed_effects, deal_id, asin, day, y]
			# r1_rating, r1_hrPred, r1_help, r1_ratio, r1_tokStd, r1_tokMean, r1_numTokens, r1_numChar, \
			# r2_rating, r2_hrPred, r2_help, r2_ratio, r2_tokStd, r2_tokMean, r2_numTokens, r2_numChar, \
			# r3_rating, r3_hrPred, r3_help, r3_ratio, r3_tokStd, r3_tokMean, r3_numTokens, r3_numChar, \
			# r4_rating, r4_hrPred, r4_help, r4_ratio, r4_tokStd, r4_tokMean, r4_numTokens, r4_numChar, \
			# r5_rating, r5_hrPred, r5_help, r5_ratio, r5_tokStd, r5_tokMean, r5_numTokens, r5_numChar]
csvfiles = os.listdir(dealData)

[c, numrev, avgrat, actualdis, dealdis, timeRem, no_type, recordtime, fixed_effects, deal_id, asin, day, y] = x_values(csvfiles, diff_Time, dealData, metaData)
	# r1_rating, r1_hrPred, r1_help, r1_ratio, r1_tokStd, r1_tokMean, r1_numTokens, r1_numChar, \
	# r2_rating, r2_hrPred, r2_help, r2_ratio, r2_tokStd, r2_tokMean, r2_numTokens, r2_numChar, \
	# r3_rating, r3_hrPred, r3_help, r3_ratio, r3_tokStd, r3_tokMean, r3_numTokens, r3_numChar, \
	# r4_rating, r4_hrPred, r4_help, r4_ratio, r4_tokStd, r4_tokMean, r4_numTokens, r4_numChar, \
	# r5_rating, r5_hrPred, r5_help, r5_ratio, r5_tokStd, r5_tokMean, r5_numTokens, r5_numChar] 

print('No. of observations - ', len(y))

x.append(c)
x.append(numrev)
x.append(avgrat)
x.append(actualdis)
x.append(dealdis)
x.append(timeRem)
# x.append(r1_rating)
# x.append(r1_hrPred)
# x.append(r1_help)
# x.append(r1_ratio)
# x.append(r1_tokStd)
# x.append(r1_tokMean)
# x.append(r1_numTokens)
# x.append(r1_numChar)
# x.append(r2_rating)
# x.append(r2_hrPred)
# x.append(r2_help)
# x.append(r2_ratio)
# x.append(r2_tokStd)
# x.append(r2_tokMean)
# x.append(r2_numTokens)
# x.append(r2_numChar)
# x.append(r3_rating)
# x.append(r3_hrPred)
# x.append(r3_help)
# x.append(r3_ratio)
# x.append(r3_tokStd)
# x.append(r3_tokMean)
# x.append(r3_numTokens)
# x.append(r3_numChar)
# x.append(r4_rating)
# x.append(r4_hrPred)
# x.append(r4_help)
# x.append(r4_ratio)
# x.append(r4_tokStd)
# x.append(r4_tokMean)
# x.append(r4_numTokens)
# x.append(r4_numChar)
# x.append(r5_rating)
# x.append(r5_hrPred)
# x.append(r5_help)
# x.append(r5_ratio)
# x.append(r5_tokStd)
# x.append(r5_tokMean)
# x.append(r5_numTokens)
# x.append(r5_numChar)
x.append(no_type)
x.append(recordtime)
x.append(fixed_effects)
x.append(day)
# x.append(ones)
# x.append(interaction)
# x.append(recordtime)
# x.append(z)

x = np.array(x)
x = x.T

## Remove no_type and record time from line 137, 138 since those are categorical variables, their std is coming out to be 0
# x[:,0:46] = (x[:,0:46] - np.mean(x[:,0:46], axis=0, dtype = np.float64)) / np.std(x[:,0:46], axis=0, dtype = np.float64)

xx.append(deal_id)
xx.append(asin)
xx = np.array(xx)
xx = xx.T

# print('VIF Factor')
# print(variance_inflation_factor(X.values, i) for i in range(X.shape[1]))
# etdata = x.set_index(['claim','number of reviews', 'average rating', 'actual discount', 'deal discount', 'time remaining'])
y = np.array(y)
y = np.expand_dims(y, axis = 1)
# y = (y - np.mean(y, axis=0, dtype = np.float64)) / np.std(y, axis=0, dtype = np.float64)
# Tried converting x and y to pandas dataframe still not working

col = ['claim', 'num_rev', 'avg_rat', 'actualdis', 'dealdis', 'timeRem', \
		# 'r1_rating', 'r1_hrPred', 'r1_help', 'r1_ratio', 'r1_tokStd', 'r1_tokMean', 'r1_numTokens', 'r1_numChar', \
		# 'r2_rating', 'r2_hrPred', 'r2_help', 'r2_ratio', 'r2_tokStd', 'r2_tokMean', 'r2_numTokens', 'r2_numChar', \
		# 'r3_rating', 'r3_hrPred', 'r3_help', 'r3_ratio', 'r3_tokStd', 'r3_tokMean', 'r3_numTokens', 'r3_numChar', \
		# 'r4_rating', 'r4_hrPred', 'r4_help', 'r4_ratio', 'r4_tokStd', 'r4_tokMean', 'r4_numTokens', 'r4_numChar', \
		# 'r5_rating', 'r5_hrPred', 'r5_help', 'r5_ratio', 'r5_tokStd', 'r5_tokMean', 'r5_numTokens', 'r5_numChar', \
		'num_type', 'recordtime', 'fixedEffects', 'day']

df = pd.DataFrame(x, columns = col)

colx = ['deal_id', 'asin']
dfx = pd.DataFrame(xx, columns = colx)
df = df.join(dfx)

coly = ['y']
dfy = pd.DataFrame(y, columns = coly)
df = df.join(dfy)
df.to_hdf(base_dir + 'No_normalized_data.h5', key = 'df', mode = 'w')

# OLS with fixed effects
# mod = PanelOLS(dfy, df, entity_effects=True)
# res = mod.fit()
# print(res)

## Simple OLS and linear regression with fixed effects
# est = sm.ols(formula = 'y ~ claim + num_rev + avg_rat + actualdis + dealdis + num_type + claim*num_rev + claim*avg_rat + claim*actualdis + claim*dealdis + timeRem + recordtime + C(fixedEffects)', data = df).fit()
# print()
# print('time difference - ', diff_Time/3600000, ' hours')
# print()
# print(est.summary())
# reg = LinearRegression().fit(x, y)

# print()
# print('regression score - ', reg.score(x, y))
# print()
# print('claim_begin', 'num_rev', 'avgRat', 'actualDis', 'dealDis', 'timeRem', 'num_type', 'Time of the day')
# print(reg.coef_)