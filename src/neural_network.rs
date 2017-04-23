use utils::{calculate_error, HaltCondition, iter_zip_enum, modified_dotprod, sigmoid};
use time::PreciseTime;
use rand::{self, Rng};
use trainer::Trainer;
use serde_json;

/// Neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Vec<Vec<f64>>>,
    num_inputs: u32,
}

impl NeuralNetwork {
    /// Each number in the `layers_sizes` parameter specifies a
    /// layer in the network. The number itself is the number of nodes in that
    /// layer. The first number is the input layer, the last
    /// number is the output layer, and all numbers between the first and
    /// last are hidden layers. There must be at least two layers in the network.
    pub fn new(layers_sizes: &[u32]) -> NeuralNetwork {
        let mut rng = rand::thread_rng();

        if layers_sizes.len() < 2 {
            panic!("must have at least two layers");
        }

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            }
        }


        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();
        // get the first layer size
        let first_layer_size = *it.next().unwrap();

        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_size {
                let mut node: Vec<f64> = Vec::new();
                for _ in 0..prev_layer_size + 1 {
                    let random_weight: f64 = rng.gen_range(-0.5f64, 0.5f64);
                    node.push(random_weight);
                }
                node.shrink_to_fit();
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        NeuralNetwork {
            layers: layers,
            num_inputs: first_layer_size,
        }
    }

    /// Runs the network on an input and returns a vector of the results.
    /// The number of `f64`s in the input must be the same
    /// as the number of input nodes in the network. The length of the results
    /// vector will be the number of nodes in the output layer of the network.
    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() as u32 != self.num_inputs {
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

    /// Takes in vector of examples and returns a `Trainer` struct that is used
    /// to specify options that dictate how the training should proceed.
    /// No actual training will occur until the `go()` method on the
    /// `Trainer` struct is called.
    pub fn train<'b>(&'b mut self, examples: &'b [(Vec<f64>, Vec<f64>)]) -> Trainer {
        Trainer::new(examples, self)
    }

    /// Encodes the network as a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("encoding JSON failed")
    }

    /// Builds a new network from a JSON string.
    pub fn from_json(encoded: &str) -> NeuralNetwork {
        let network: NeuralNetwork = serde_json::from_str(encoded).expect("decoding JSON failed");
        network
    }

    pub fn train_details(&mut self,
                         examples: &[(Vec<f64>, Vec<f64>)],
                         rate: f64,
                         momentum: f64,
                         log_interval: Option<u32>,
                         halt_condition: HaltCondition)
                         -> f64 {

        // check that input and output sizes are correct
        let input_layer_size = self.num_inputs;
        let output_layer_size = self.layers[self.layers.len() - 1].len();
        for &(ref inputs, ref outputs) in examples.iter() {
            if inputs.len() as u32 != input_layer_size {
                panic!("input has a different length than the network's input layer");
            }
            if outputs.len() != output_layer_size {
                panic!("output has a different length than the network's output layer");
            }
        }

        self.train_incremental(examples, rate, momentum, log_interval, halt_condition)
    }

    fn train_incremental(&mut self,
                         examples: &[(Vec<f64>, Vec<f64>)],
                         rate: f64,
                         momentum: f64,
                         log_interval: Option<u32>,
                         halt_condition: HaltCondition)
                         -> f64 {

        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0u32;
        let mut training_error_rate = 0f64;
        let start_time = PreciseTime::now();

        loop {
            if epochs > 0 {
                // log error rate if necessary
                match log_interval {
                    Some(interval) if epochs % interval == 0 => {
                        println!("error rate: {}", training_error_rate);
                    }
                    _ => (),
                }

                // check if we've met the halt condition yet
                match halt_condition {
                    HaltCondition::Epochs(epochs_halt) => {
                        if epochs == epochs_halt {
                            break;
                        }
                    }
                    HaltCondition::MSE(target_error) => {
                        if training_error_rate <= target_error {
                            break;
                        }
                    }
                    HaltCondition::Timer(duration) => {
                        let now = PreciseTime::now();
                        if start_time.to(now) >= duration {
                            break;
                        }
                    }
                }
            }

            training_error_rate = 0f64;

            for &(ref inputs, ref targets) in examples.iter() {
                let results = self.do_run(&inputs);
                let weight_updates = self.calculate_weight_updates(&results, &targets);
                training_error_rate += calculate_error(&results, &targets);
                self.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
            }

            epochs += 1;
        }

        training_error_rate
    }

    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push(sigmoid(modified_dotprod(&node, &results[layer_index])))
            }
            results.push(layer_results);
        }
        results
    }

    // updates all weights in the network
    fn update_weights(&mut self,
                      network_weight_updates: &Vec<Vec<Vec<f64>>>,
                      prev_deltas: &mut Vec<Vec<Vec<f64>>>,
                      rate: f64,
                      momentum: f64) {
        for layer_index in 0..self.layers.len() {
            let mut layer = &mut self.layers[layer_index];
            let layer_weight_updates = &network_weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let mut node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                    let delta = (rate * weight_update) + (momentum * prev_delta);
                    node[weight_index] += delta;
                    prev_deltas[layer_index][node_index][weight_index] = delta;
                }
            }
        }

    }

    // calculates all weight updates by backpropagation
    fn calculate_weight_updates(&self,
                                results: &Vec<Vec<f64>>,
                                targets: &[f64])
                                -> Vec<Vec<Vec<f64>>> {
        let mut network_errors: Vec<Vec<f64>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.layers;
        let network_results = &results[1..]; // skip the input layer
        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in
            iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();


            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();
                let node_error;

                // calculate error for this node
                if layer_index == layers.len() - 1 {
                    node_error = result * (1f64 - result) * (targets[node_index] - result);
                } else {
                    let mut sum = 0f64;
                    let next_layer_errors = &network_errors[network_errors.len() - 1];
                    for (next_node, &next_node_error_data) in
                        next_layer_nodes.unwrap().iter().zip((next_layer_errors).iter()) {
                        sum += next_node[node_index + 1] * next_node_error_data; // +1 because the 0th weight is the threshold
                    }
                    node_error = result * (1f64 - result) * sum;
                }

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    let prev_layer_result;
                    if weight_index == 0 {
                        prev_layer_result = 1f64; // threshold
                    } else {
                        prev_layer_result = prev_layer_results[weight_index - 1];
                    }
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // updates were built by backpropagation so reverse them
        network_weight_updates.reverse();

        network_weight_updates
    }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }

        network_level
    }
}
