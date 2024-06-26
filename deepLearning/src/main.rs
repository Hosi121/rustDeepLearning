mod fastforward_network;
mod activation_functions;
mod initialize;
use rand::Rng;

fn main() {
    let number_layers = 3;
    let number_neurons = vec![2, 3, 1];
    let input = vec![0.0, 1.0];
    let input_size = input.len();
    let (weights, biases) = initialize::initialize(number_layers, &number_neurons, input_size);
    let result = fastforward_network::fastforward_network(number_layers, number_neurons, input, weights, biases);
    println!("{:?}", result);
}