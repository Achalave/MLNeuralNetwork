package mlneuralnetwork;

import java.util.Arrays;

//@author Michael Haertling
public class Perceptron {

    private final double[] weights;
    private final ActivationFunction function;
    private final double threshold;

    public Perceptron(int numInputs, double thresh, ActivationFunction func) {
        weights = new double[numInputs];
        Arrays.fill(weights, 0);
        function = func;
        threshold = thresh;
    }

    public double getDotProduct(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    public double getRawOutput(double[] inputs) {
        return function.activate(getDotProduct(inputs));
    }

    public double getBoundedOutput(double[] inputs) {
        return (getRawOutput(inputs) >= threshold) ? 1 : 0;
    }

    public double[] getWeights() {
        return weights;
    }

}
