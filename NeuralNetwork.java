import java.util.Random;

class Neuron {
  private Random random;
  private double[] weights;
  private double bias;
  private double[] inputs;
  private double result;
  private double delta;

  public double initialize() {
    return this.random.nextDouble() / 10 + .01;
  }

  public Neuron(int size, Random random) {
    this.random = random;

    this.weights = new double[size];
    for (int i=0; i<size; i++)
      this.weights[i] = this.initialize();

    this.bias = this.initialize();
  }

  public double forwardPass(double[] inputs) {
    if (inputs.length != this.weights.length)
      throw new IllegalArgumentException();

    this.inputs = inputs;
    double sum = this.bias;
    for (int i=0; i<inputs.length; i++)
      sum += this.weights[i] * inputs[i];
    return this.result = activationFunction(sum);
  }

  public double getResult() {
    return this.result;
  }

  public double activationFunction(double input) {
    return input > 0 ? input : input / 10;
  }

  public double derivative() {
    return this.result > 0 ? 1 : 0.1;
  }

  public double[] getWeights() {
    return this.weights.clone();
  }

  public double getBias() {
    return this.bias;
  }

  public double getDelta() {
    return this.delta;
  }

  public double setDelta(double delta) {
    double old = this.delta;
    this.delta = delta;
    return old;
  }

  public void updateParameters(double learningRate) {
    double k = learningRate * this.delta * this.derivative();

    for (int i = 0; i < this.weights.length; i++) {
      this.weights[i] -= k * this.inputs[i];
      if (Double.isNaN(this.weights[i]))
        this.weights[i] = this.initialize();
    }
    
    this.bias -= k;
    if (Double.isNaN(this.bias))
      this.bias = this.initialize();
  }
}

class Layer {
  private Neuron[] neurons;
  private double[] results;

  public Layer(int numberOfNeurons, int size, Random random) {
    this.neurons = new Neuron[numberOfNeurons];
    for (int i=0; i<numberOfNeurons; i++)
      this.neurons[i] = new Neuron(size, random);

    this.results = new double[numberOfNeurons];
  }

  public double[] forwardPass(double[] inputs) {
    for (int i=0; i<this.neurons.length; i++)
      this.results[i] = this.neurons[i].forwardPass(inputs);

    return this.results.clone();
  }

  public Neuron[] getNeurons() {
    return this.neurons.clone();
  }

  public void updateParameters(double learningRate) {
    for (Neuron neuron : this.neurons)
      neuron.updateParameters(learningRate);
  }
}

class Network {
  private Layer[] layers;

  public Network(int[] sizes, Random random) {
    layers = new Layer[sizes.length-1]; 
    for (int i=0; i<this.layers.length; i++)
      layers[i] = new Layer(sizes[i+1], sizes[i], random);
  }

  public double[] forwardPass(double[] inputs) {
    for (int i=0; i<this.layers.length; i++)
      inputs = this.layers[i].forwardPass(inputs);

    return inputs;
  }

  public Layer[] getLayers() {
    return this.layers.clone();
  }

  public void updateParameters(double learningRate) {
    for (Layer layer : this.layers)
      layer.updateParameters(learningRate);
  }

  public void backPropagate(double[] expected) {
    for (int i = layers.length - 1; i >= 0; i--) {
      Layer layer = layers[i];

      if (i == layers.length - 1) {
        for (int j = 0; j < layer.getNeurons().length; j++) {
          Neuron neuron = layer.getNeurons()[j];
          neuron.setDelta(neuron.getResult() - expected[j]);
        }
        continue;
      }

      Layer nextLayer = layers[i + 1];
      for (int j = 0; j < layer.getNeurons().length; j++) {
        Neuron neuron = layer.getNeurons()[j];
        double error = 0;
        for (Neuron nextNeuron : nextLayer.getNeurons())
          error += nextNeuron.getWeights()[j] * nextNeuron.getDelta();

        neuron.setDelta(error);
      }
    }
  }

  public void train(
    double[][] inputs, 
    double[][] expected, 
    double learningRate
  ) {
    for (int i=0; i<inputs.length; i++) {
      this.forwardPass(inputs[i]);
      this.backPropagate(expected[i]);
      this.updateParameters(learningRate);
    }
  }
}

public class NeuralNetwork {
  public static void main(String[] args) {
    Random random = new Random(42);

    int[] layerSizes = {1, 5, 3, 1};
    Network network = new Network(layerSizes, random);

    double learningRate = 1;
    for (int t = 0; t < 10; t++) {
      for (int i = 0; i < 3000000; i++) {
        double[][] inputs = new double[1][1];
        double[][] expectedOutputs = new double[1][1];
        double input = random.nextDouble() * Math.PI / 2;
        double expectedOutput = Math.sin(input);
        inputs[0][0] = input;
        expectedOutputs[0][0] = expectedOutput;
        network.train(inputs, expectedOutputs, learningRate);
      }
      learningRate /= 10;
    }

    for (double i = 0; i <= Math.PI / 2; i+=0.0002) {
      double[] inputs = new double[1];
      double input = i;
      inputs[0] = input;
      double expectedOutput = Math.sin(input);
      double output = network.forwardPass(inputs)[0];
      System.out.printf("%.4f\t%.4f\n", input, output);
    }
  }
}
