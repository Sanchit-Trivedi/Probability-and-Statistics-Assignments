import pandas as pd
import copy
def spect():
	'''This function applies naive bayes classification on the given data set spect and reports the accuracy'''
	# Defining training and testing data
	training = pd.read_csv("SPECT.train",names = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22'],sep=',')
	testing = pd.read_csv("SPECT.test",names = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22'],sep=',')
	test_len = len(testing)
	train_len = len(training)
	total_0 = len(training[training['y']==0])
	total_1 = len(training[training['y']==1])
	cols = list(training)
	correct = 0 #no. of correct predictions
	#Carrying out predictions and comparison with test data
	for i in range(test_len):
		output = testing.at[i,'y']
		prob0 = 1;prob1 = 1
		for j in range(1,len(cols)):
			test=testing.at[i,cols[j]]
			num0 = len(training[((training['y'])==0) & (training[cols[j]]==test)])
			num1 = len(training[((training['y'])==1) & (training[cols[j]]==test)])
			prob0*=(num0/total_0)
			prob1*=(num1/total_1)
		if (prob0*(total_0/train_len)>prob1*(total_1/train_len) and output==0):
			correct+=1
		elif (prob0*(total_0/train_len)<prob1*(total_1/train_len) and output==1):
			correct+=1
	print("SPECT Data Set Accuracy : ",(correct/test_len)*100)

def monks():
	# tuple of all datasets for monks
	files = ("monks-1.train","monks-1.test","monks-2.train","monks-2.test","monks-3.train","monks-3.test")
	c=0
	for i in range(0,len(files),2):
		#defining training and testing
		training = pd.read_csv(files[i],names = ['y','x1','x2','x3','x4','x5','x6','x7'],sep=" ")
		testing = pd.read_csv(files[i+1],names = ['y','x1','x2','x3','x4','x5','x6','x7'],sep=" ")
		testing = testing.drop(['x7'],axis=1)
		training = training.drop(['x7'],axis=1)
		test_len = len(testing)
		train_len = len(training)
		total_0 = len(training[training['y']==0])
		total_1 = len(training[training['y']==1])
		# print(total_0,total_1)
		cols = list(training)
		correct = 0 #counter for correct predictions
		for i in range(test_len):
			output = testing.iloc[i,0]
			prob0 = 1; prob1 = 1;
			for j in range(1,len(cols)):
				test = testing.iloc[i,j]
				num0 = len(training[((training['y'])==0) & (training[cols[j]]==test)])
				num1 = len(training[((training['y'])==1) & (training[cols[j]]==test)])
				prob0*=(num0/total_0)
				prob1*=(num1/total_1)
			if (prob0*(total_0/train_len)>prob1*(total_1/train_len) and output==0):
				correct+=1
			elif (prob0*(total_0/train_len)<prob1*(total_1/train_len) and output==1):
				correct+=1
		c+=1
		print("MONKS Data Set "+str(c)+" Accuracy : ",(correct/test_len)*100)

def shuttle():
	''' This function uses leave one out cross validation of naive bayes classification'''
	data = pd.read_csv("shuttle-landing-control.data", sep=",", header=None)# reading the data
	y=len(list(data))
	data[y]=-1
	n=len(data)
	true=[True for i in range(n)]
	autofactor=len(data[data[0]==2])/n # Final factors with which probabilty has to be multiplied i.e P(y)
	noautofactor=1-autofactor 
	for i in range(n):
		boolean=copy.deepcopy(true)
		boolean[i]=False # list for leave one out cross validation
		training=data[boolean]
		autoprob=1;noautoprob=1
		for j in range(1,y-1):
			testing=data.at[i,j]
			if testing!='*': # handling don't care
				auto=training[training[0]==2]
				autoprob*=len(auto[auto[j].isin([testing,'*'])])/len(auto) # probability for auto 
				noauto=training[training[0]==1]
				noautoprob*=len(noauto[noauto[j].isin([testing,'*'])])/len(noauto) #probability for noauto
		if autoprob*autofactor>noautoprob*noautofactor:
			data.at[i,y]=1
		elif autoprob*autofactor<noautoprob*noautofactor:
			data.at[i,y]=2
	correctlen=len(data[data[0]==data[y]])
	# data.to_csv("OUTPUT3_1.txt",index=False,header=None)
	print("Shuttle Landing Data Set Accuracy : ",(correctlen*100/n)) # printing the output

def soyabean():
	''' This function uses leave one out cross validation of naive bayes classification'''
	data = pd.read_csv("soybean-small.data", sep=",", header=None)
	y=len(list(data))
	data[y]="none"
	n=len(data)
	true=[True for i in range(n)]
	#P(y) for final multiplication
	d1factor=len(data[data[y-1]=="D1"])/n
	d2factor=len(data[data[y-1]=="D2"])/n
	d3factor=len(data[data[y-1]=="D3"])/n
	d4factor=len(data[data[y-1]=="D4"])/n
	for i in range(n):
		boolean=copy.deepcopy(true)
		boolean[i]=False
		training=data[boolean] # leaving one out
		d1prob=1;d2prob=1;d3prob=1;d4prob=1;
		for j in range(y-2):
			testing=data.at[i,j]
			# probabilities for D1, d2,d3 ,d4
			D1=training[training[y-1]=="D1"]
			d1prob*=len(D1[D1[j]==testing])/len(D1)
			D2=training[training[y-1]=="D2"]
			d2prob*=len(D2[D2[j]==testing])/len(D2)
			D3=training[training[y-1]=="D3"]
			d3prob*=len(D3[D3[j]==testing])/len(D3)
			D4=training[training[y-1]=="D4"]
			d4prob*=len(D4[D4[j]==testing])/len(D4)
		l=[d1prob*d1factor,d2prob*d2factor,d3prob*d3factor,d4prob*d4factor]
		if l!=[0.0,0.0,0.0,0.0]:
			data.at[i,y]="D"+str(l.index(max(l))+1) # Assigning predictions
	# data.to_csv("OUTPUT2.csv",index=False)
	correctlen=(len(data[data[y-1]==data[y]])-1)
	print("Soybean Data Set Accuracy : ",(correctlen*100/n))

def tictactoe():
	''' This function uses leave one out cross validation of naive bayes classification'''
	data = pd.read_csv("tic-tac-toe.data", sep=",", header=None) # reading the data
	data[10]="none"
	n=len(data)
	true=[True for i in range(n)]
	# P(y) for final multiplication
	posfactor=len(data[data[9]=="positive"])/n
	negfactor=1-posfactor
	for i in range(n):
		boolean=copy.deepcopy(true)
		boolean[i]=False
		training=data[boolean] # leaving one out and the rest as training
		posprob=1;negprob=1
		for j in range(len(list(data))-2):
			testing=data.at[i,j]
			pos=training[training[9]=="positive"]
			posprob*=len(pos[pos[j]==testing])/len(pos)
			neg=training[training[9]=="negative"]
			negprob*=len(neg[neg[j]==testing])/len(neg)
		# Assigning predictions
		if posprob*posfactor>negprob*negfactor:
			data.at[i,10]="positive"
		elif posprob*posfactor<negprob*negfactor:
			data.at[i,10]="negative"
	correctlen=(len(data[data[9]==data[10]])-4)
	# data.to_csv("OUTPUT.csv",index=False)
	print("Tic-Tac-Toe Data Set Accuracy : ",(correctlen*100/n))

spect()
monks()
shuttle()
soyabean()
tictactoe()