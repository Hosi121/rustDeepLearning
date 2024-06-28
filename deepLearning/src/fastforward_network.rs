use crate::activation_functions;

pub fn forward_network(
    number_layers: usize,
    number_neurons: Vec<usize>,
    input: Vec<f64>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut output = input.clone();
    let mut layer_outputs = vec![output.clone()];
    let mut layer_inputs = vec![];

    for l in 0..number_layers {
        let mut new_output = Vec::new();
        let mut layer_input = Vec::new();
        for j in 0..number_neurons[l] {
            let mut sum = biases[l][j];
            for i in 0..output.len() {
                sum += output[i] * weights[l][j][i];
            }
            layer_input.push(sum);
            new_output.push(activation_functions::sigmoid(sum));
        }
        layer_inputs.push(layer_input);
        output = new_output.clone();
        layer_outputs.push(output.clone());
    }

    (layer_outputs, layer_inputs)
}

pub fn backpropagation(
    number_layers: usize,
    number_neurons: Vec<usize>,
    input: Vec<f64>,
    weights: &mut Vec<Vec<Vec<f64>>>,
    biases: &mut Vec<Vec<f64>>,
    expected_output: Vec<f64>,
    learning_rate: f64
) -> Vec<f64> {
    // 順伝播
    let (layer_outputs, layer_inputs) = forward_network(number_layers, number_neurons.clone(), input, weights.clone(), biases.clone());

    // 誤差の初期化
    let mut deltas = vec![vec![0.0; number_neurons[number_layers - 1]]];

    // 出力層の誤差計算
    for j in 0..number_neurons[number_layers - 1] {
        let output = layer_outputs[number_layers][j];
        deltas[0][j] = (output - expected_output[j]) * activation_functions::sigmoid_derivative(layer_inputs[number_layers - 1][j]);
    }

    // 隠れ層の誤差計算
    for l in (0..number_layers - 1).rev() {
        let mut new_deltas = vec![0.0; number_neurons[l]];
        for i in 0..number_neurons[l] {
            let mut error = 0.0;
            for j in 0..number_neurons[l + 1] {
                error += deltas[number_layers - l - 2][j] * weights[l + 1][j][i];
            }
            new_deltas[i] = error * activation_functions::sigmoid_derivative(layer_inputs[l][i]);
        }
        deltas.push(new_deltas);
    }
    deltas.reverse();

    // 重みとバイアスの更新
    for l in 0..number_layers {
        for j in 0..number_neurons[l] {
            for i in 0..layer_outputs[l].len() {
                weights[l][j][i] -= learning_rate * deltas[l][j] * layer_outputs[l][i];
            }
            biases[l][j] -= learning_rate * deltas[l][j];
        }
    }

    layer_outputs[number_layers].clone()
}
