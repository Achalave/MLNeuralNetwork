/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlneuralnetwork;

import java.io.FileNotFoundException;
import java.text.DecimalFormat;

/**
 *
 * @author Michael
 */
public class MLNeuralNetwork {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException {
        //training file
        //test file
        //learning rate
        //number of iterations
        
        if(args.length!=4){
            System.out.println("Must provide four arguments as follows:\n"
                    + "(1) training file path\n"
                    + "(2) test file path\n"
                    + "(3) learning rate (double value)\n"
                    + "(4) number of iterations (integer value)");
            return;
        }
        
        String trainingFile = args[0];
        String testFile = args[1];
        double learningRate = Double.parseDouble(args[2]);
        int numIterations = Integer.parseInt(args[3]);
        
//        String trainingFile = "C:\\Users\\Michael\\Google Drive\\School\\UTD Year 4\\Intro To Machine Learning\\HW2 - Neural Networks\\Part 2\\train2.dat";
//        String testFile = "C:\\Users\\Michael\\Google Drive\\School\\UTD Year 4\\Intro To Machine Learning\\HW2 - Neural Networks\\Part 2\\test2.dat";
//        double learningRate = 0.9;
//        int numIterations = 800;
        
        GraidientDecentTrainer trainer = new GraidientDecentTrainer(trainingFile, learningRate, numIterations);
        Perceptron p = trainer.train(true);
        
        //Test the perceptron
        //Test over training set
        DecimalFormat df = new DecimalFormat(".0");
//        df.setRoundingMode(RoundingMode.FLOOR);
        Dataset data = GraidientDecentTrainer.readDataFile(trainingFile);
        System.out.println("\nAccuracy on training set ("+data.getNumInstances()+" instances): "+df.format(test(p,data.data)*100)+"%");
        
        //Test over test set
        data = GraidientDecentTrainer.readDataFile(testFile);
        System.out.println("\nAccuracy on test set ("+data.getNumInstances()+" instances): "+df.format(test(p,data.data)*100)+"%");
    }
    
    public static double test(Perceptron p, double[][] data){
        int numCorrect = 0;
        for(double[] instance:data){
            double output = p.getBoundedOutput(instance);
            double correctOutput = instance[instance.length-1];
            if(correctOutput == output){
                numCorrect++;
            }
        }
        return (double)numCorrect/data.length;
    }
    
}
