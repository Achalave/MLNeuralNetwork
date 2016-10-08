package mlneuralnetwork;

import java.util.Arrays;

//@author Michael Haertling
public class Dataset {

    public final double[][] data;
    public final String[] header;

    public Dataset(String[] h, double[][] data) {
        this.header = h;
        this.data = data;
    }

    public int getInstanceLength() {
        return header.length;
    }

    public int getNumInstances() {
        return data.length;
    }

    @Override
    public String toString() {
        String out = Arrays.toString(header) + "\n";
        for (double[] instance : data) {
            out += Arrays.toString(instance)+"\n";
        }
        return out;
    }

}
