import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.core.converters.ArffSaver;

import java.io.*;
import java.util.*;

/**
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AddAttribute {
  /**
   * adds the attributes
   *
   * @param args    the commandline arguments
   */
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.out.println("\nUsage: <file.arff> <filter|java>\n");
      System.exit(1);
    }

    // load dataset
    Instances data = new Instances(new BufferedReader(new FileReader(args[0])));
    Instances newData = null;

    // filter or java?
    if (args[1].equals("filter")) {
      Add filter;
      newData = new Instances(data);
      filter = new Add();
      filter.setAttributeIndex("last");
      filter.setAttributeName("revenue");
      filter.setInputFormat(newData);
      newData = Filter.useFilter(newData, filter);
    }
    else if (args[1].equals("java")) {
      newData = new Instances(data);
      // add new attributes
      // 1. nominal

      newData.insertAttributeAt(new Attribute("revenue"), newData.numAttributes());
    }
    else {
      System.out.println("\nUsage: <file.arff> <filter|java>\n");
      System.exit(2);
    }

    // random values
    Random rand = new Random(1);
    for (int i = 0; i < newData.numInstances(); i++) {

      newData.instance(i).setValue(newData.numAttributes() - 1, (int)(rand.nextDouble()*100000));
    }

    // output on stdout
    System.out.println(newData);
    ArffSaver saver = new ArffSaver();
    saver.setInstances(newData);
    saver.setFile(new File(args[0]));
    saver.writeBatch();
  }
}
