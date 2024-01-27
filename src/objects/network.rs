use std::error;
use std::fmt;

use ndarray::Array1;

use crate::objects::layer::NetworkLayer;

type DynResult<T> = std::result::Result<T, Box<dyn error::Error>>;

#[derive(Debug, Clone)]
pub struct AlreadyInitializedError;
impl fmt::Display for AlreadyInitializedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "This operation is not allowed on an initialized network")
    }
}
impl error::Error for AlreadyInitializedError {}

#[derive(Debug, Clone)]
pub struct NotInitializedError;
impl fmt::Display for NotInitializedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "This operation is not allowed on an uninitialized network"
        )
    }
}
impl error::Error for NotInitializedError {}

#[derive(Debug, Clone)]
pub struct EmptyNetworkError;
impl fmt::Display for EmptyNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "This operation is not allowed on an empty network")
    }
}
impl error::Error for EmptyNetworkError {}

pub struct Network {
    layers: Vec<Box<dyn NetworkLayer>>,
    initialized: bool,
}

impl Network {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            initialized: false,
        }
    }

    pub fn add_layer(
        &mut self,
        layer: Box<dyn NetworkLayer>,
    ) -> Result<(), AlreadyInitializedError> {
        if self.initialized {
            return Err(AlreadyInitializedError);
        } else {
            self.layers.push(layer);
            Ok(())
        }
    }

    pub fn initilize(&mut self, input_size: usize) -> DynResult<()> {
        if self.layers.len() == 0 {
            return Err(EmptyNetworkError.into());
        } else if self.initialized == true {
            return Err(AlreadyInitializedError.into());
        } else {
            let mut layer_input_size = input_size;
            for i in 0..self.layers.len() {
                self.layers[i].initilize(layer_input_size);
                layer_input_size = self.layers[i].get_neurons();
            }
            self.initialized = true;
            Ok(())
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn get_output_size(&self) -> Option<usize> {
        if self.initialized {
            Some(self.layers[self.layers.len() - 1].get_neurons())
        } else {
            None
        }
    }

    /// Returns the input size of the network
    /// Returns None if the network is not initialized
    pub fn get_input_size(&self) -> Option<usize> {
        self.layers[0].get_input_size()
    }

    /// Evaluates the network with the given input
    /// panics if the network is not initialized
    pub fn evaluate(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output = input.clone();
        for i in 0..self.layers.len() {
            output = self.layers[i].evaluate(&output);
        }
        output
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::objects::layer::{activation_functions, FullyConnectedLayer};

    #[test]
    fn test_initialization() {
        let mut network = Network::new();
        let layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        assert_eq!(network.is_initialized(), false);
        assert_eq!(layer.initialized(), false);
        network.add_layer(Box::new(layer)).unwrap();
        network.initilize(2).unwrap();
        assert_eq!(network.is_initialized(), true);
        assert_eq!(network.layers[0].initialized(), true);
    }

    #[test]
    fn test_empty_initialization() {
        let mut network = Network::new();
        let res = network.initilize(2);
        assert_eq!(network.initialized, false);
        assert!(res.is_err());
        println!("{:?}", res.err().unwrap());
    }

    #[test]
    fn test_already_initialized() {
        let mut network = Network::new();
        let layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        network.add_layer(Box::new(layer)).unwrap();
        network.initilize(2).unwrap();
        let res = network.initilize(2);
        assert_eq!(network.initialized, true);
        assert!(res.is_err());
    }

    #[test]
    fn test_initialization_control() {
        let mut network = Network::new();
        let layer = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        network.add_layer(Box::new(layer)).unwrap();
        network.initilize(2).unwrap();
        let layer2 = FullyConnectedLayer::new(3, activation_functions::Sigmoid);
        assert_eq!(network.add_layer(Box::new(layer2)).is_err(), true);
        assert_eq!(network.initilize(2).is_err(), true);
    }

    #[test]
    fn test_network() {
        let mut network = Network::new();
        let layer1 = FullyConnectedLayer::new(3, activation_functions::Nothing);
        let layer2 = FullyConnectedLayer::new(2, activation_functions::Nothing);
        let layer3 = FullyConnectedLayer::new(1, activation_functions::Nothing);
        network.add_layer(Box::new(layer1)).unwrap();
        network.add_layer(Box::new(layer2)).unwrap();
        network.add_layer(Box::new(layer3)).unwrap();
        // network.initilize_with_distribution(2, &TestDistribution).unwrap();
        network.initilize(2).unwrap();
        let input = Array1::from(vec![0.5, 0.5]);
        let output = network.evaluate(input);
        assert_eq!(output.shape()[0], network.get_output_size().unwrap());
        // assert_eq!(output, Array1::from(vec![15.0]));
    }
}
