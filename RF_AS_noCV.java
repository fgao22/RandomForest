import weka.core.converters.*;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.filters.unsupervised.attribute.Discretize;
import weka.classifiers.Evaluation;

public class RestaurantRevenue {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("train.csv");
		Instances dataset = source.getDataSet();
	
		AttributeSelection filter = new AttributeSelection();
		//create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		//set the filter to use the evaluator and search algorithm
		filter.setEvaluator(eval);
		filter.setSearch(search);
		//specify the dataset
		filter.setInputFormat(dataset);
		//apply
		Instances filteredData = Filter.useFilter(dataset, filter);

		String[] options = new String[4];
		options[0] = "-F";
		options[1] = "-R"; options[2] = "last";
		options[3] = "-V";

		Discretize discretize = new Discretize();
		discretize.setOptions(options);
		discretize.setInputFormat(filteredData);
		Instances discretizedData = Filter.useFilter(filteredData, discretize);

		discretizedData.setClassIndex(discretizedData.numAttributes()-1);
		//create and build the classifier!
		RandomForest rf = new RandomForest();
		rf.buildClassifier(discretizedData);
		//print out capabilities
		Evaluation evaluation = new Evaluation(discretizedData);
		evaluation.evaluateModel(rf, discretizedData);
		System.out.println(evaluation.toSummaryString("Evaluation results:\n", false));

		ArffSaver saver = new ArffSaver();
		saver.setInstances(discretizedData);
		saver.setFile(new File("proccessedData.arff"));
		saver.writeBatch();
	}
}