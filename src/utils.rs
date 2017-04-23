use std::iter::{Zip, Enumerate};
use time::Duration;
use std::slice;

pub const DEFAULT_LEARNING_RATE: f64 = 0.3f64;
pub const DEFAULT_MOMENTUM: f64 = 0f64;
pub const DEFAULT_EPOCHS: u32 = 1000;

/// Specifies when to stop training the network
#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    /// Stop training after a certain number of epochs
    Epochs(u32),
    /// Train until a certain error rate is achieved
    MSE(f64),
    /// Train for some fixed amount of time and then halt
    Timer(Duration),
}

/// Specifies which [learning mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to use when training the network
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    /// train the network Incrementally (updates weights after each example)
    Incremental,
}

pub fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

pub fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + (-y).exp())
}


// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
pub fn iter_zip_enum<'s, 't, S: 's, T: 't>
    (s: &'s [S],
     t: &'t [T])
     -> Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>> {
    s.iter().zip(t.iter()).enumerate()
}

// calculates MSE of output layer
pub fn calculate_error(results: &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    let ref last_results = results[results.len() - 1];
    let mut total: f64 = 0f64;
    for (&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f64)
}
