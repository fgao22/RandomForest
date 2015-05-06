Original data: train.csv test.csv

Data Preprocessing:

	1. Change date to survival dates and convert restaurant types so that we get
	
	train_final.csv test_final.csv

	2. Convert csv file to arff file by running:
	
	javac -cp ".:weka.jar" CSV2Arff.java
	java -cp ".:weka.jar" CSV2Arff train_final.csv train_final.arff
	java -cp ".:weka.jar" CSV2Arff test_final.csv test_final.arff
	
	3. Add the class column "revenue" to the test_final set and populate it with random numbers
	by running:
	
	javac -cp ".:weka.jar" AddAttribute.java
	java -cp ".:weka.jar" AddAttribute filter
	
	The actual numbers are not important (details later)
	
Running the Random Forest classification by command line from weka
	
	java -cp weka.jar weka.classifiers.misc.InputMappedClassifier -W weka.classifiers.trees.randomForest -t train_final.arff -T test_final.arff -classifications weka.classifiers.evaluation.output.prediction.CSV > submission.csv
	
	This command uses weka library to train the random fortest model with the train_final set,
	and make predictions by the information of the test_final set. It will return "submission.csv"
	that contains an Id, an actual value (which are the random numbers we generated), a predicted value
	(what we really care about) and the error (not important)
	
	Note: 
	1) For Cities attributes, there are more cities in the test set than in the train set, thus causing an
	imcompatibility issue in weka. the InputMappedClassifier we used in command line handles this
	by assuming a missing value if there is value that was not seen in the model
	2) -W The classifier we are using to train the model
	   -t train set
	   -T test set
	3) weka.classifiers.evaluation.output.prediction.CSV output the predictions to a CSV formatt
	
Data Postprossessing
	
	For submission.csv, we got ride of the actual attribute and the error attribute, decrement all
	IDs by one so that it starts from 0 (Kaggle requirement)

