use rand::Rng;

pub fn initialize(
    number_layers: usize,
    number_neurons: &Vec<usize>,
    input_size: usize
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();

    let mut weights = Vec::new();
    for l in 0..number_layers {
        let mut layer_weights = Vec::new();
        let layer_input_size = if l == 0 { input_size } else { number_neurons[l - 1] };
        for _ in 0..number_neurons[l] {
            let mut neuron_weights = Vec::new();
            for _ in 0..layer_input_size {
                neuron_weights.push(rng.gen_range(-1.0..1.0));
            }
            layer_weights.push(neuron_weights);
        }
        weights.push(layer_weights);
    }
    let mut biases = Vec::new();
    for &num in number_neurons {
        let mut layer_biases = Vec::new();
        for _ in 0..num {
            layer_biases.push(rng.gen_range(-1.0..1.0));
        }
        biases.push(layer_biases);
    }

    (weights, biases)
}