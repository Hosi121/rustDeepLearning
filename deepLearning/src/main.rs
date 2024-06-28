mod fastforward_network;
mod activation_functions;
mod initialize;
use rand::Rng;

fn main() {
    let number_layers = 3;
    let number_neurons = vec![2, 3, 1];
    let input = vec![0.0, 1.0];
    let expected_output = vec![1.0];
    let input_size = input.len();
    let learning_rate = 0.1;

    let (mut weights, mut biases) = initialize::initialize(number_layers, &number_neurons, input_size);

    for _ in 0..100000 { // ループ数は適宜調整
        fastforward_network::backpropagation(number_layers, number_neurons.clone(), input.clone(), &mut weights, &mut biases, expected_output.clone(), learning_rate);
    }

    let result = fastforward_network::forward_network(number_layers, number_neurons, input, weights, biases);
    println!("Final Output: {:?}", result.0.last().unwrap());
}
