package mlneuralnetwork;



//@author Michael Haertling

public class SigmoidActivationFunction implements ActivationFunction{

    double response;
    
    public SigmoidActivationFunction(){
        response = 1;
    }
    
    @Override
    public double activate(double in) {
        return (1/(1+Math.exp(-in/response)));        
    }
    
}
