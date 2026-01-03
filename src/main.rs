use ndarray::{self, Array1, Array2};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use regex::Regex;
use std::fs::File;
use std::io::prelude::*;

const NB_PIXELS: usize = 784; // Px per image
const NB_LAYER_1: usize = 64;
const NB_LAYER_2: usize = 32;
const NB_LAYER_OUTPUT: usize = 10;

fn extract_next_number(matches: &mut regex::Matches) -> Option<u16> {
    if let Some(matched) = matches.next() {
        if let Ok(num) = matched.as_str().parse::<u16>() {
            return Some(num);
        }
    }
    None
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone)]
struct Image {
    pixels: Array2<u16>,
    label: u16,
}

impl Image {
    fn new() -> Image {
        Image {
            pixels: Array2::zeros((28, 28)),
            label: 0,
        }
    }
}

struct Layer {
    neurons: Array1<f64>,
    weights_per_neuron: Array2<f64>,
    pre_activation: Array1<f64>,
    biases: Array1<f64>,
    weight_gradient: Array2<f64>,
    bias_gradient: Array1<f64>,
}

impl Layer {
    fn new(nb_neurons: usize, neurons_previous_layer: usize) -> Layer {
        let mut rng = thread_rng();
        let scale = (2.0 / (neurons_previous_layer as f64 + nb_neurons as f64)).sqrt();

        Layer {
            neurons: Array1::zeros(nb_neurons),
            pre_activation: Array1::zeros(nb_neurons),
            weights_per_neuron: Array2::from_shape_fn((nb_neurons, neurons_previous_layer), |_| {
                rng.gen_range(-scale..scale)
            }),
            biases: Array1::from_shape_fn(nb_neurons, |_| rng.gen_range(-0.1..0.1)),
            weight_gradient: Array2::zeros((nb_neurons, neurons_previous_layer)),
            bias_gradient: Array1::zeros(nb_neurons),
        }
    }
}
struct Network {
    input_layer: Layer,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
}

impl Network {
    fn new() -> Network {
        let hidden_layer1 = Layer::new(NB_LAYER_1, NB_PIXELS);
        let hidden_layer2 = Layer::new(NB_LAYER_2, NB_LAYER_1);
        Network {
            input_layer: Layer::new(NB_PIXELS, 0),
            hidden_layers: vec![hidden_layer1, hidden_layer2],
            output_layer: Layer::new(NB_LAYER_OUTPUT, NB_LAYER_2),
        }
    }

    fn input_images(&mut self, data: &Image) {
        let normalized = data.pixels.mapv(|x| x as f64 / 255.0);
        self.input_layer.neurons = normalized.into_shape(NB_PIXELS).unwrap();
    }

    fn compute(&mut self) -> (){
        self.hidden_layers[0].pre_activation = self.hidden_layers[0]
            .weights_per_neuron
            .dot(&self.input_layer.neurons)
            + &self.hidden_layers[0].biases;
        self.hidden_layers[0].neurons = self.hidden_layers[0].pre_activation.mapv(|x| sigmoid(x));

        self.hidden_layers[1].pre_activation = self.hidden_layers[1]
            .weights_per_neuron
            .dot(&self.hidden_layers[0].neurons)
            + &self.hidden_layers[1].biases;
        self.hidden_layers[1].neurons = self.hidden_layers[1].pre_activation.mapv(|x| sigmoid(x));

        self.output_layer.pre_activation = self
            .output_layer
            .weights_per_neuron
            .dot(&self.hidden_layers[1].neurons)
            + &self.output_layer.biases;
        self.output_layer.neurons = self.output_layer.pre_activation.mapv(|x| sigmoid(x));
    }
    fn return_answer(&self) -> usize {
        self.output_layer
            .neurons
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn cost_func(&self, label: u16) -> f64 {
        let mut target = Array1::zeros(10);
        target[label as usize] = 1.0;

        let epsilon = 1e-10;
        -target
            .iter()
            .zip(self.output_layer.neurons.iter())
            .map(|(t, &o)| t * (o.max(epsilon)).ln())
            .sum::<f64>()
    }

    fn backpropagate(&mut self, label: u16, learning_rate: f64) {
        let mut target = Array1::zeros(NB_LAYER_OUTPUT);
        target[label as usize] = 1.0;

        let output_error = &self.output_layer.neurons - &target;

        let output_sigmoid_deriv = self.output_layer.neurons.mapv(|x| x * (1.0 - x));
        let output_delta = &output_error * &output_sigmoid_deriv;

        self.output_layer.weight_gradient = output_delta.view().insert_axis(ndarray::Axis(1)).dot(
            &self.hidden_layers[1]
                .neurons
                .view()
                .insert_axis(ndarray::Axis(0)),
        );

        self.output_layer.bias_gradient = output_delta.clone();

        let hidden2_error = self.output_layer.weights_per_neuron.t().dot(&output_delta);
        let hidden2_sigmoid_deriv = self.hidden_layers[1].neurons.mapv(|x| x * (1.0 - x));
        let hidden2_delta = &hidden2_error * &hidden2_sigmoid_deriv;

        self.hidden_layers[1].weight_gradient =
            hidden2_delta.view().insert_axis(ndarray::Axis(1)).dot(
                &self.hidden_layers[0]
                    .neurons
                    .view()
                    .insert_axis(ndarray::Axis(0)),
            );

        self.hidden_layers[1].bias_gradient = hidden2_delta.clone();

        let hidden1_error = self.hidden_layers[1]
            .weights_per_neuron
            .t()
            .dot(&hidden2_delta);
        let hidden1_sigmoid_deriv = self.hidden_layers[0].neurons.mapv(|x| x * (1.0 - x));
        let hidden1_delta = &hidden1_error * &hidden1_sigmoid_deriv;

        self.hidden_layers[0].weight_gradient =
            hidden1_delta.view().insert_axis(ndarray::Axis(1)).dot(
                &self
                    .input_layer
                    .neurons
                    .view()
                    .insert_axis(ndarray::Axis(0)),
            );

        self.hidden_layers[0].bias_gradient = hidden1_delta.clone();

        self.update_weights(learning_rate);
    }

    fn update_weights(&mut self, learning_rate: f64) {
        self.hidden_layers[0].weights_per_neuron = &self.hidden_layers[0].weights_per_neuron
            - learning_rate * &self.hidden_layers[0].weight_gradient;
        self.hidden_layers[0].biases =
            &self.hidden_layers[0].biases - learning_rate * &self.hidden_layers[0].bias_gradient;

        self.hidden_layers[1].weights_per_neuron = &self.hidden_layers[1].weights_per_neuron
            - learning_rate * &self.hidden_layers[1].weight_gradient;
        self.hidden_layers[1].biases =
            &self.hidden_layers[1].biases - learning_rate * &self.hidden_layers[1].bias_gradient;

        self.output_layer.weights_per_neuron = &self.output_layer.weights_per_neuron
            - learning_rate * &self.output_layer.weight_gradient;
        self.output_layer.biases =
            &self.output_layer.biases - learning_rate * &self.output_layer.bias_gradient;
    }
}

fn create_batches(data: &[Image], batch_size: usize) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut rng = thread_rng();

    indices.shuffle(&mut rng);

    let mut batches = Vec::new();
    for chunk in indices.chunks(batch_size) {
        batches.push(chunk.to_vec());
    }

    batches
}

fn main() {
    let image_n = 6000;
    let learning_rate = 0.1;
    let epochs = 30;
    let batch_size = 32;

    println!("Loading MNIST dataset...");

    let mut data: Vec<Image> = Vec::new();
    let mut file: File =
        File::open("mnist_handwritten/mnist_handwritten_train.json").expect("Unable to read file.");
    let mut contents: String = String::new();

    let s: String = String::from("image");
    file.read_to_string(&mut contents)
        .expect("Couldn't put contents of the file in the variable contents.");

    let c: usize = contents.matches(&s).count();
    let re: Regex = Regex::new(r"\d+").unwrap();
    let mut a = re.find_iter(&contents);

    let max_images = image_n.min(c);
    println!("Parsing {} images...", max_images);

    for i in 1..=max_images {
        let mut current_image: Image = Image::new();
        for j in 1..=28 {
            for k in 1..=28 {
                current_image.pixels[[j - 1, k - 1]] =
                    extract_next_number(&mut a).expect("Error parsing pixel");
            }
        }
        current_image.label = extract_next_number(&mut a).expect("Couldn't find the next number.");
        data.push(current_image);

        if i == image_n {
            println!("  Loaded {} images", i);
        }
    }

    println!("Loaded {} images total\n", data.len());

    let mut batches = create_batches(&data, batch_size);
    println!("Created {} batches of size {}\n", batches.len(), batch_size);

    let mut neural_network = Network::new();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut sample_count = 0;
        let mut epoch_correct = 0;

        let mut rng = thread_rng();
        batches.shuffle(&mut rng);

        for batch_indices in &batches {
            for &idx in batch_indices {
                let image = &data[idx];

                neural_network.input_images(image);
                neural_network.compute();

                let loss = neural_network.cost_func(image.label);
                total_loss += loss;
                sample_count += 1;

                let prediction = neural_network.return_answer();
                if prediction == image.label as usize {
                    epoch_correct += 1;
                }

                neural_network.backpropagate(image.label, learning_rate);
            }
        }

        let avg_loss = total_loss / sample_count as f64;
        let epoch_accuracy = (epoch_correct as f64 / sample_count as f64) * 100.0;

        println!(
            "Epoch {:2}/{}: Loss = {:.6}, Accuracy = {:.2}%",
            epoch + 1,
            epochs,
            avg_loss,
            epoch_accuracy
        );
    }

    println!("\n--- Training Complete ---\n");

    println!("\nLoading MNIST test set...");

    let mut test_data: Vec<Image> = Vec::new();
    let mut test_file: File = File::open("mnist_handwritten/mnist_handwritten_test.json")
        .expect("Unable to read test file.");
    let mut test_contents: String = String::new();

    test_file
        .read_to_string(&mut test_contents)
        .expect("Couldn't read test file contents.");

    let test_count = test_contents.matches("image").count();
    let test_re: Regex = Regex::new(r"\d+").unwrap();
    let mut test_iter = test_re.find_iter(&test_contents);

    let max_test_images = test_count;
    println!("Parsing {} test images...", max_test_images);

    for i in 1..=max_test_images {
        let mut img = Image::new();

        for j in 1..=28 {
            for k in 1..=28 {
                img.pixels[[j - 1, k - 1]] =
                    extract_next_number(&mut test_iter).expect("Error parsing test pixel");
            }
        }

        img.label = extract_next_number(&mut test_iter).expect("Error parsing test label");

        test_data.push(img);

        if i == image_n {
            println!("  Loaded {} test images", i);
        }
    }
    println!("Evaluating on test set...");
    let mut test_correct = 0;
    let mut test_class_correct = vec![0; 10];
    let mut test_class_total = vec![0; 10];

    for image in &test_data {
        neural_network.input_images(image);
        neural_network.compute();
        let prediction = neural_network.return_answer();

        let label = image.label as usize;
        test_class_total[label] += 1;

        if prediction == label {
            test_correct += 1;
            test_class_correct[label] += 1;
        }
    }

    let test_accuracy = (test_correct as f64 / test_data.len() as f64) * 100.0;

    println!(
        "Test Accuracy: {:.2}% ({}/{})\n",
        test_accuracy,
        test_correct,
        test_data.len()
    );

    println!("Per-class Test Accuracy:");
    for digit in 0..10 {
        if test_class_total[digit] > 0 {
            let class_acc =
                (test_class_correct[digit] as f64 / test_class_total[digit] as f64) * 100.0;
            println!(
                "  Digit {}: {:.2}% ({}/{})",
                digit, class_acc, test_class_correct[digit], test_class_total[digit]
            );
        }
    }
}
