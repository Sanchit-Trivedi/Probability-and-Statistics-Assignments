# Name : Sanchit Trivedi
# Roll No. : 2018091
# Section : A
# Maths Assignment 1

import os,stat,sys
import pandas,math

def texttocsv():
	''' This function converts the given found.txt into a csv file for easier processing'''

	if os.path.isfile("WithoutHI.csv"):os.chmod("WithoutHI.csv",stat.S_IRWXU)
	#Converting found.txt into a csv file containing the Impact Factors
	f=open("found.txt","r")
	f1=open("WithoutHI.csv","w+") 
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
	os.chmod("WithoutHI.csv",777)

def merging():
	'''This function finds matches between the file download from the internet i.e WithoutIF and the file obtained from found.txt
		It then writes the Matched data to a file "Output.csv " '''

	data1= pandas.read_csv('WithoutHI.csv',header=None,names=["Title","H index","Impact Factor"],usecols=["Title","Impact Factor"])#Reading data form  csv files
	data2= pandas.read_csv("WithoutIF.csv",sep=";",usecols=["Title","H index","Type"])
	data1["Title"]=data1.Title.str.replace(" ","").str.replace(",","").str.lower()#Removing special characters for comparison and merging
	data2["Title"]=data2.Title.str.replace(" ","").str.replace(",","").str.lower()
	data3=pandas.merge(data1,data2,on="Title")
	data3.to_csv("Output.csv",index=False)#Writing the merged output to a csv file

def computing():
	''' This function computes the correlation coefficents, Linear Regression coefficients and Mean Squared Error Between the Predicted and Actual Values of the Impact factor for Journals
		It also computes the Impact Factor Values for Conferences Based On findings from Journals'''

	sys.stdout=open("Computed_Values.txt","w")#Writing the output to text file
	print("The computed values are :")
	print()
	data=pandas.read_csv('Output.csv')
	traininglen=int(len(data)*0.8)#Defining Training data
	predictionlen=len(data)-traininglen#Defining Prediction data

	training=data.iloc[:traininglen]
	prediction=data.iloc[traininglen:]

	x=training["H index"]#x values
	y=training["Impact Factor"]#y values

	meanx=sum(x)/traininglen #Finding the mean
	meany=sum(y)/traininglen

	varx=(sum(x**2)/traininglen)-meanx**2 #Finding the variance
	vary=(sum(y**2)/traininglen)-meany**2

	stdx=math.sqrt(varx) # Finding the standard deviation
	stdy=math.sqrt(vary)

	covariance=(sum(x*y)/traininglen)-meanx*meany # Finding covariance

	r=covariance/(stdx*stdy)#Finding the correlation coefficient
	print("The Correlation Coefficient between H Index and Impact Factor is : ",r)
	print()
	
	# Finding the regression coefficients
	a=r*stdy/stdx
	b=meany-a*meanx
	print("The Regression Coefficients A and B of the form AX+B are :")
	print ("A: ",a," B: ",b)
	print()

	predictedif = a*prediction["H index"] + b #Predicted Impact Factor Values using Linear Regression
	error=predictedif-prediction["Impact Factor"]#Calculating the error
	errorsq=error**2# Error Squared
	meansqerror=sum(errorsq)/predictionlen # Mean Squared Error
	print("The Mean Square error: ",meansqerror)
	print()

	conf=pandas.read_csv("conferences.csv",sep=";",usecols=["Title","H index","Type"])# Reading the Conference data downloaded from the internet
	conf["Impact Factor"]=a*conf["H index"]+b # Predicting Impact Factors for Conferences using values obtained from Journals
	conf.to_csv("ConferencesFinal.csv",index=False)#Writing the output to the file
	print("The Predicted Impact Factors for conferences are stored in file: ConferencesFinal.csv ")

def sum(x):
	'''This function compute the sum of given iterable object x'''

	s=0
	for i in x: 
		s+=i
	return s

texttocsv()
merging()
computing()