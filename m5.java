import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class m5 {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("pre.csv");
		Instances dataset = source.getDataSet();
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		AttributeSelection filter = new AttributeSelection();
		
		//create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();				
		BestFirst search = new BestFirst();
		
		//set the filter to use the evaluator and search algorithm
		filter.setEvaluator(eval);
		filter.setSearch(search);
		
		//specify the dataset
		filter.setInputFormat(dataset);
		
		//apply
		Instances filteredData = Filter.useFilter(dataset, filter);
		//System.out.println(filteredData.toSummaryString());
		
		//create and build the classifier!
		M5P m5 = new M5P();
		m5.buildClassifier(filteredData);
		
		//print out capabilities
		Evaluation evaluation = new Evaluation(filteredData);
		evaluation.crossValidateModel(m5, filteredData, 10, new Random(1));
		System.out.println(evaluation.toSummaryString("Evaluation results:\n", false));
		
	}
}
