use crate::activation_functions;

pub fn fastforward_network(number_layers: usize, number_neurons: Vec<usize>, input: Vec<f64>, weights: Vec<Vec<Vec<f64>>>, biases: Vec<Vec<f64>>) -> Vec<f64> {
    let mut output = input.clone();
    for l in 0..number_layers {
        let mut new_output = Vec::new();
        for j in 0..number_neurons[l] {
            let mut sum = biases[l][j];
            for i in 0..output.len() {
                sum += output[i] * weights[l][j][i];
            }
            new_output.push(activation_functions::sigmoid(sum));
        }
        output = new_output;
    }
    output
}
