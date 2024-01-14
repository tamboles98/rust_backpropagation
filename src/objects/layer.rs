use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub mod activation_functions {
/// Represents an activation function
    pub trait Activate {
        /// Evaluates the activation function
        fn activate(&self, input: f32) -> f32;
    }

    struct Sigmoid;
    impl Activate for Sigmoid {
        fn activate(&self, input: f32) -> f32 {
            1. / (1. + (-input).exp())
        }
    }

    struct ReLU;
    impl Activate for ReLU {
        fn activate(&self, input: f32) -> f32 {
            if input > 0. {
                input
            } else {
                0.
            }
        }
    }

    struct Tanh;
    impl Activate for Tanh {
        fn activate(&self, input: f32) -> f32 {
            input.tanh()
        }
    }

    struct Nothing;
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

/// Represents a layer of a neural network
pub struct Layer<S>{ 
    weigths: Array2<f32>,
    biases: Array1<f32>,
    activation: S
}

impl <S> Layer<S> where S: activation_functions::Activate{
    /// Creates a new layer with random weights and biases
    pub fn new(neurons: usize, input_size: usize, activation: S) -> Self {
        let distr = Uniform::new(0., 1.);
        let weights = Array2::random((neurons, input_size), distr);
        let biases = Array1::random((neurons,), distr);
        Layer {
            weigths: weights,
            biases: biases,
            activation: activation
        }
    }

    pub fn initilize(&mut self) {
        let distr = Uniform::new(0., 1.);
        self.weigths = Array2::random(self.weigths.dim(), distr);
        self.biases = Array1::random(self.biases.dim(), distr);
    }

    /// Returns the number of neurons in the layer
    pub fn neurons(&self) -> usize {
        self.weigths.shape()[0]
    }

    /// Returns the number of inputs to the layer
    pub fn input_size(&self) -> usize {
        self.weigths.shape()[1]
    }

    /// Evaluates the layer with the given input
    /// !panics if the input shape is wrong
    pub fn evaluate(&self, input: &Array1<f32>) -> Array1<f32> {
        let res = self.weigths.dot(input) + &self.biases;
        res.mapv(|x| self.activation.activate(x))
    }
}
