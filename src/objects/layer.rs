use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Distribution};

pub mod activation_functions {
/// Represents an activation function
    pub trait Activate {
        /// Evaluates the activation function
        fn activate(&self, input: f32) -> f32;
    }

    pub struct Sigmoid;
    impl Activate for Sigmoid {
        fn activate(&self, input: f32) -> f32 {
            1. / (1. + (-input).exp())
        }
    }

    pub struct ReLU;
    impl Activate for ReLU {
        fn activate(&self, input: f32) -> f32 {
            if input > 0. {
                input
            } else {
                0.
            }
        }
    }

    pub struct Tanh;
    impl Activate for Tanh {
        fn activate(&self, input: f32) -> f32 {
            input.tanh()
        }
    }

    pub struct Nothing;
    impl Activate for Nothing {
        fn activate(&self, input: f32) -> f32 {
            input
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn test_sigmoid() {
            let sigmoid = Sigmoid;
            assert_eq!(sigmoid.activate(0.), 0.5);
            assert_eq!(sigmoid.activate(1.), 0.7310585786300049);
            assert_eq!(sigmoid.activate(-1.), 0.2689414213699951);
        }

        #[test]
        fn test_relu() {
            let relu = ReLU;
            assert_eq!(relu.activate(0.), 0.);
            assert_eq!(relu.activate(1.), 1.);
            assert_eq!(relu.activate(-1.), 0.);
        }

        #[test]
        fn test_tanh() {
            let tanh = Tanh;
            assert_eq!(tanh.activate(0.), 0.);
            assert_eq!(tanh.activate(1.), 0.7615941559557649);
            assert_eq!(tanh.activate(-1.), -0.7615941559557649);
        }

        #[test]
        fn test_nothing() {
            let nothing = Nothing;
            assert_eq!(nothing.activate(0.), 0.);
            assert_eq!(nothing.activate(1.), 1.);
            assert_eq!(nothing.activate(-1.), -1.);
        }
    }
}


pub trait NetworkLayer{
    fn evaluate(&self, input: &Array1<f32>) -> Array1<f32>;
    fn initilize(&mut self, input_size: usize);
    fn initialized(&self) -> bool;
    fn get_input_size(&self) -> Option<usize>;
    fn get_neurons(&self) -> usize;
}

/// Represents a layer of a neural network
pub struct FullyConnectedLayer<S> { 
    neurons: usize,
    weights: Option<Array2<f32>>,
    biases: Array1<f32>,
    activation: S
}

impl <S> FullyConnectedLayer<S>
where S: activation_functions::Activate{
    /// Creates a new unitialized layer with the given number of neurons and
    /// activation function
    pub fn new(neurons: usize, activation: S) -> Self {
        let distr = Uniform::new(0., 1.);
        let weights = Option::None;
        let biases = Array1::random((neurons,), distr);
        FullyConnectedLayer {
            neurons: neurons,
            weights,
            biases: biases,
            activation: activation
        }
    }

    pub fn initilize_with_distribution(&mut self, input_size: usize, distr: &impl Distribution<f32>) {
        self.weights = Option::Some(Array2::random((self.neurons, input_size), distr));
        self.biases = Array1::random(self.biases.dim(), distr);
    }
}

impl <S> NetworkLayer for FullyConnectedLayer<S>
where S: activation_functions::Activate {

    fn initialized(&self) -> bool {
        self.weights.is_some()
    }

    fn initilize(&mut self, input_size: usize) {
        let distr = Uniform::new(0., 1.);
        self.weights = Option::Some(Array2::random((self.neurons, input_size), distr));
        self.biases = Array1::random(self.biases.dim(), distr);
    }

    /// Returns the number of neurons in the layer
    fn get_neurons(&self) -> usize {
        self.neurons
    }

    /// Returns the number of inputs to the layer
    fn get_input_size(&self) -> Option<usize> {
        match self.weights.as_ref() {
            Some(array) => Some(array.shape()[1]),
            None => None
        }
    }

    /// Evaluates the layer with the given input
    /// !panics if the input shape is wrong
    /// !panics if the layer is not initialized
    fn evaluate(&self, input: &Array1<f32>) -> Array1<f32> {
        let weights = self.weights.as_ref().unwrap();
        let res = weights.dot(input) + &self.biases;
        res.mapv(|x| self.activation.activate(x))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray_rand::rand;

    struct TestDistribution;
    impl Distribution<f32> for TestDistribution {
        fn sample<R: rand::Rng + ?Sized>(&self, _: &mut R) -> f32 {
            1.
        }
    }

    #[test]
    fn test_layer() {
        let layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        assert_eq!(layer.get_neurons(), 3);
        assert_eq!(layer.get_input_size(), None);
        assert_eq!(layer.initialized(), false);
    }

    #[test]
    fn test_layer_initialized() {
        let mut layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        layer.initilize(2);
        assert_eq!(layer.get_neurons(), 3);
        assert_eq!(layer.get_input_size(), Some(2));
        assert_eq!(layer.initialized(), true);
    }

    #[test]
    fn test_layer_initialized_distribution() {
        let mut layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        layer.initilize_with_distribution(2, &TestDistribution);
        assert_eq!(layer.get_neurons(), 3);
        assert_eq!(layer.get_input_size(), Some(2));
        assert_eq!(layer.initialized(), true);
        assert_eq!(layer.weights.unwrap(),
                Array2::from_shape_vec((3, 2), vec![1., 1., 1., 1., 1., 1.])
            .unwrap());
        assert_eq!(layer.biases, Array1::from_shape_vec((3,), vec![1., 1., 1.])
            .unwrap());
    }

    #[test]
    fn test_layer_evaluate() {
        let mut layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        layer.initilize(2);
        let input = Array1::from(vec![1., 2.]);
        let output = layer.evaluate(&input);
        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_layer_evaluate_value() {
        let mut layer = FullyConnectedLayer::new(2, activation_functions::Nothing);
        layer.initilize_with_distribution(2, &TestDistribution);
        let input = Array1::from(vec![1., 2.]);
        let output = layer.evaluate(&input);
        assert_eq!(output, Array1::from(vec![4., 4.]));
    }
}