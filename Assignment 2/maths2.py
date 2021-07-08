# Name : Sanchit Trivedi
# Roll No. : 2018091
# Section : A
# Maths Assignment 2

import os,stat,sys
import pandas,math
import itertools
import numpy as np

def texttocsv():
	''' This function converts the given found.txt into a csv file for easier processing'''
	if os.path.isfile("WithIF.csv"):os.chmod("WithIF.csv",stat.S_IRWXU)
	#Converting found.txt into a csv file containing the Impact Factors
	f=open("found.txt","r")
	f1=open("WithIF.csv","w+") 
	data=f.read()
	fdata=""
	for i in range(len(data)):
			if data[i]==";":
					fdata+=","
			else:
					fdata+=data[i]
	f1.write(fdata)
	f1.close()
	f.close()
	os.chmod("WithIF.csv",777)

def merging():
	'''This function finds matches between the file download from the internet i.e WithoutIF and the file obtained from found.txt
		It then writes the Matched data to a file "Output.csv " '''
	data1= pandas.read_csv('WithIF.csv',header=None,names=["Title","H index","Impact Factor"],usecols=["Title","Impact Factor"])#Reading data from  csv files
	data2= pandas.read_csv("WithoutIF.csv",sep=";")
	data2["SJR"]=data2.SJR.str.replace(",",".")
	data2["Cites / Doc. (2years)"]=data2["Cites / Doc. (2years)"].str.replace(",",".")
	data2["Ref. / Doc."]=data2["Ref. / Doc."].str.replace(",",".")
	data1["Title"]=data1.Title.str.replace(" ","").str.replace(",","").str.lower()#Removing special characters for comparison and merging
	data2["Title"]=data2.Title.str.replace(" ","").str.replace(",","").str.lower()
	data3=pandas.merge(data2,data1,on="Title")
	data3.sort_values(by=["Rank"],inplace=True)
	data3=data3.dropna()
	data3.to_csv("Merged.csv",index=False)#Writing the merged output to a csv file

def computation():
	
	''' This function computes the Linear Regression coefficients, Mean Absolute Error and Mean Squared Error Between the Predicted and Actual Values of the Impact factor for every possible combination 
	of Independent Variables and stores them in the file Output.csv'''

	data=pandas.read_csv('Merged.csv')
	data = data.drop(["Country","Publisher","Categories","SJR Best Quartile","Issn","Type","Sourceid","Rank"],axis=1)
	col=list(data)
	col.remove("Title")
	col.remove("Impact Factor")
	comb=[]
	for x in range(2,len(col)+1):
		comb.extend(list(map(list,itertools.combinations(col,x))))

	traininglen=int(len(data)*0.8)#Defining Training data
	predictionlen=len(data)-traininglen#Defining Prediction data
	training=data.iloc[:traininglen]
	prediction=data.iloc[traininglen:]

	'''Finds the regression coefficients'''
	dfarr=[]
	for i in comb:
		temp=training[i]
		ones = np.ones([traininglen,1])
		temp= np.concatenate((temp,ones),axis=1)
		trans=temp.transpose()
		mul=np.matmul(trans,temp)
		mulinv=np.linalg.inv(mul)
		mat=np.matmul(mulinv,trans)
		dfarr.append(np.matmul(mat,training["Impact Factor"].values))
	
	'''Finds the predicted values of Impact Factor using regression coefficients'''	
	predictedarr=[]
	for j in range(len(dfarr)):
		temp = prediction[comb[j]]
		ones = np.ones([predictionlen,1])
		temp = np.concatenate((temp,ones),axis=1)
		predicted=np.matmul(temp,dfarr[j])
		predictedarr.append(pandas.DataFrame(predicted,index=None,columns=["PredictedIF"]))
	
	''' Finding the errors'''
	errorlist=[]
	sqerrorlist=[]
	for i in range(len(predictedarr)):
		sum1 = 0;sum2=0
		error=predictedarr[i]["PredictedIF"]-prediction["Impact Factor"]
		for j in range(len(predictedarr[i]["PredictedIF"])):
			sum2 +=abs(list(predictedarr[i]["PredictedIF"])[j]- list(prediction["Impact Factor"])[j])
			sum1 += (list(predictedarr[i]["PredictedIF"])[j]- list(prediction["Impact Factor"])[j])**2
		sum1 = sum1/len(list(predictedarr[i]["PredictedIF"]))
		sum2 = sum2/len(list(predictedarr[i]["PredictedIF"]))
		sqerrorlist.append(sum1)
		errorlist.append(sum2)

	towrite={
		'Combinations':comb, 
        'Mean Squared Error':sqerrorlist, 
        'Mean Absolute Error':errorlist 
        }
	df=pandas.DataFrame(towrite)
	df.sort_values(by=["Mean Squared Error"],inplace=True)
	df.to_csv("Output.csv",index=False)
	pandas.options.display.max_colwidth = 200
	print()
	print ("Combination with minimum Mean Squared Error")
	print()
	print(df.loc[df['Mean Squared Error'].idxmin()])
	print()
	print ("Combination with minimum Mean Absolute Error")
	print()
	print(df.loc[df['Mean Absolute Error'].idxmin()])
	print()
	print("					All combination data has been stored in file Output.csv")

texttocsv()
merging()
computation()