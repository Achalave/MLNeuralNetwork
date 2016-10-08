package mlneuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

//@author Michael Haertling
public class GraidientDecentTrainer {

    double learningRate;
    int numIterations;
    Dataset data;
    Perceptron perc;

    private final DecimalFormat df = new DecimalFormat("0.0000");
    
    public GraidientDecentTrainer(String trainingFile, double learningRate, int numIterations) throws FileNotFoundException {
        this.learningRate = learningRate;
        this.numIterations = numIterations;

        //Read in the file
        data = GraidientDecentTrainer.readDataFile(trainingFile);
        
//        System.out.println(data);

        //Create the perceptron to train
        perc = new Perceptron(data.getInstanceLength() - 1, 0.5, new SigmoidActivationFunction());
    }

    public static Dataset readDataFile(String filePath) throws FileNotFoundException {
        double[][] out;
        Scanner in = new Scanner(new File(filePath));
        String header = null;
        //Find the header line
        while (header == null) {
            String tmp = in.nextLine();
            //Ignore lines with spaces
            if (!tmp.contains(" ")) {
                header = tmp;
            }
        }

        ArrayList<double[]> data = new ArrayList<>();

        while (in.hasNextLine()) {
            String tmp = in.nextLine();
            //Skip lines with spaces
            if (tmp.contains(" ")) {
                continue;
            }
            //Split the array
            String[] tmpArray = tmp.split("\t");
            //Cast the array to an int[]
            double[] finalArray = new double[tmpArray.length];
            for (int i = 0; i < tmpArray.length; i++) {
                finalArray[i] = Double.parseDouble(tmpArray[i]);
            }
            //Add the array to the dataset
            data.add(finalArray);
        }

        //Place the data in the static array variable and return it
        out = new double[data.size()][];
        out = data.toArray(out);
        return new Dataset(header.split("\t"), out);
    }

    public Perceptron train(boolean print) {
        int index = 0;
        int iteration = 1;
        while (numIterations > 0) {
//            System.out.println("train "+iteration);
            //Do training
            trainOverInstance(index);
            //Print is enabled
            if (print) {
                double[] weights = perc.getWeights();
                String printOut = "After iteration " + iteration + ": ";
                //Print all the new weights
                for (int i = 0; i < weights.length; i++) {
                    printOut += "w(" + data.header[i] + ") = "+df.format(weights[i])+", ";
                }
                //Print the new output
                printOut += "output = "+df.format(perc.getRawOutput(data.data[index]));
                System.out.println(printOut);
            }
            //Increment/decrement counters
            numIterations--;
            index = (index == data.getNumInstances() - 1) ? 0 : index + 1;
            iteration++;
        }

        return perc;
    }

    public void trainOverInstance(int index) {
        double[] instance = data.data[index];
        //Get the output
        double output = perc.getRawOutput(instance);
//        System.out.println("Output: "+output);
        //Get the expected output
        double expectedOutput = instance[instance.length - 1];
//        System.out.println("Expected Output: "+expectedOutput);
        //Get the error
        double error = expectedOutput - output;
        //Update the weights
        //w <- w + adx(sigma derivitive(w*x))
        double[] weights = perc.getWeights();
        for (int i = 0; i < weights.length; i++) {
            //w        = w          + a            * d     * sigma der.   * x           * output
            weights[i] = weights[i] + learningRate * error * (1 - output) * instance[i] * output;
        }
    }
}
