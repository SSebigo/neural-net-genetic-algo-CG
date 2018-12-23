use rand::Rng;

#[derive(Debug, Clone)]
struct Neuron {
    output_value: f64,
    output_weights: Vec<f64>,
    weight_index: usize,
}

#[derive(Debug, Clone)]
struct Network {
    layers: Vec<Vec<Neuron>>,
}

#[derive(Debug, Clone)]
struct Individual {
    neural_network: Network,
    fitness_score: u32,
}

#[derive(Debug)]
struct GeneticAlgorithm {
    elitism_rate: f64,
    mutation_rate: f64,
    population_size: u32,
    topology: Vec<u32>,
    population: Vec<Individual>,
}

impl Neuron {
    fn new(nbr_outputs: usize, weight_index: usize) -> Neuron {

        let mut output_weights: Vec<f64> = Vec::new();
        for _ in 0..nbr_outputs {
            output_weights.push(rand::thread_rng().gen::<f64>());
        }

        Neuron { output_value: 0.0_f64, output_weights: output_weights, weight_index: weight_index }
    }

    fn feed_forward(&mut self, prev_layer: &Vec<Neuron>) {

        let mut sum: f64 = 0.0_f64;

        for neuron_nbr in 0..prev_layer.len() {
            sum += (prev_layer[neuron_nbr].output_value * prev_layer[neuron_nbr].output_weights[self.weight_index]) as f64;
        }

        self.output_value = self.tanh(sum);
    }

    fn tanh(&self, v: f64) -> f64 {
        v.tanh()
    }
}

impl Network {
    fn new(topology: &Vec<u32>) -> Network {

        let nbr_layers: usize = topology.len();

        let mut layers: Vec<Vec<Neuron>> = Vec::new();
        for layer_nbr in 0..nbr_layers {

            layers.push(Vec::new());

            let nbr_outputs: usize = if layer_nbr == topology.len()-1 {
                0
            } else {
                topology[layer_nbr+1] as usize
            };

            println!("Layer created!");
            for neuron_nbr in 0..topology[layer_nbr]+1 {

                layers[layer_nbr].push(Neuron::new(nbr_outputs, neuron_nbr as usize));
                println!("Neuron created!");
            }

            let bias_index = layers[layer_nbr].len()-1;
            layers[layer_nbr][bias_index].output_value = 1.0;
        }

        Network { layers: layers }
    }

    fn create_children_network(neurons: Vec<Vec<Neuron>>) -> Network {

        Network { layers: neurons }
    }

    fn feed_forward(&mut self, inputs: &Vec<f64>) {

        assert_eq!(inputs.len(), self.layers[0].len()-1);

        for neuron_nbr in 0..inputs.len() {
            self.layers[0][neuron_nbr].output_value = inputs[neuron_nbr];
        }

        for layer_nbr in 1..self.layers.len() {

            let prev_layer: Vec<Neuron> = self.layers[layer_nbr-1].clone();
            for neuron_nbr in 0..self.layers[layer_nbr].len()-1 {
                self.layers[layer_nbr][neuron_nbr].feed_forward(&prev_layer);
            }
        }
    }

    fn print_network(&self) {
        println!("{:#?}", self);
    }
}

impl Individual {
    fn new(topology: &Vec<u32>) -> Individual {

        Individual {
            neural_network: Network::new(topology),
            fitness_score: 0,
        }
    }

    fn create_children(neurons: Vec<Vec<Neuron>>) -> Individual {

        Individual {
            neural_network: Network::create_children_network(neurons),
            fitness_score: 0,
        }
    }
}

impl GeneticAlgorithm {
    fn new() -> GeneticAlgorithm {

        let elitism_rate: f64 = 0.0_f64;
        let mutation_rate: f64 = 0.0_f64;

        let population_size: u32 = 100;

        let topology: Vec<u32> = vec![2, 4, 2];

        let mut population: Vec<Individual> = Vec::new();
        for _pop in 0..population_size {
            population.push(Individual::new(&topology))
        }

        GeneticAlgorithm {
            elitism_rate: elitism_rate,
            mutation_rate: mutation_rate,
            population_size: population_size,
            topology: topology,
            population: population,
        }
    }

    fn training(&mut self, training_data: &Vec<(Vec<u32>, u32)>, elitism_rate: f64, mutation_rate: f64) {
        self.elitism_rate = elitism_rate;
        self.mutation_rate = mutation_rate;

        for indiv in &mut self.population {
            for td in training_data {
                let inputs: Vec<f64> = vec![1.0_f64/td.0[0] as f64, 1.0_f64/td.0[1] as f64];
                indiv.neural_network.feed_forward(&inputs);
                let outputs: Vec<f64> = vec![indiv.neural_network.layers[self.topology.len()-1][0].output_value,
                                                indiv.neural_network.layers[self.topology.len()-1][1].output_value];
                let prediction: usize = if outputs[0] > outputs[1] { 0 } else { 1 };
                if td.1 == td.0[prediction] {
                    indiv.fitness_score += 1;
                }
            }
        }
    }

    fn selection(&mut self) -> f64 {
        let breeding_pop: u32 = ((self.population_size as f64) * self.elitism_rate) as u32;
        // println!("breeding_pop: {}", breeding_pop);
        self.population.sort_by_key(|x| x.fitness_score);
        self.population.reverse();

        let best_score: f64 = (self.population[0].fitness_score as f64/1000.0_f64)*100.0_f64;

        let mut index: u32 = 0;
        for indiv in &mut self.population {
            if index == 10 { break; };
            index += 1;
            println!("Training results: {}/1000 -> {}% correct", indiv.fitness_score, (indiv.fitness_score as f64/1000.0_f64)*100.0_f64);
        }

        let mut new_pop: Vec<Individual> = Vec::new();
        for i in 0..breeding_pop {
            new_pop.push(self.population.remove(0));
            new_pop[i as usize].fitness_score = 0;
        }
        println!("New pop created!");
        self.population.clear();
        self.population = new_pop;

        best_score
    }

    fn breeding(&mut self) {

        let nbr_children: u32 = self.population_size -
                                (((self.population_size as f64) * self.elitism_rate) as f64) as u32;
        // println!("nbr children: {}", nbr_children);
        let pop_len: usize = self.population.len();
        // println!("population len: {}", pop_len);
        for _ in 0..(nbr_children/2) {

            let father_index: usize = rand::thread_rng().gen_range(0, pop_len);
            let mut mother_index: usize = rand::thread_rng().gen_range(0, pop_len);
            if mother_index == father_index { mother_index = rand::thread_rng().gen_range(0, pop_len); }

            let mut father: Individual = self.population[father_index].clone();
            let mut mother: Individual = self.population[mother_index].clone();

            let mut child1_neurons: Vec<Vec<Neuron>> = Vec::new();
            let mut child2_neurons: Vec<Vec<Neuron>> = Vec::new();

            for i in 0..3 {
                child1_neurons.push(Vec::new());
                child2_neurons.push(Vec::new());

                assert_eq!(father.neural_network.layers[i].len(), mother.neural_network.layers[i].len());
                for j in 0..(father.neural_network.layers[i].len()) {
                    if j <= (father.neural_network.layers[i].len()-1)/2 {
                        child1_neurons[i].push(father.neural_network.layers[i].remove(0));
                        child2_neurons[i].push(mother.neural_network.layers[i].remove(0));
                    } else {
                        child1_neurons[i].push(mother.neural_network.layers[i].remove(0));
                        child2_neurons[i].push(father.neural_network.layers[i].remove(0));
                    }
                }
            }

            self.population.push(Individual::create_children(child1_neurons));
            self.population.push(Individual::create_children(child2_neurons));
        }
        // println!("population len: {}", self.population.len());
    }

    fn mutation(&mut self) {

        let mutation_nbr: u32 = (16.0_f64 * self.mutation_rate) as u32;
        println!("mutation_nbr: {}", mutation_nbr);

        for indiv in &mut self.population {
            for _ in 0..mutation_nbr {

                let mutate_layer: usize = rand::thread_rng().gen_range(0, self.topology.len()-1);
                let mutate_neuron: usize = rand::thread_rng().gen_range(0, indiv.neural_network.layers[mutate_layer].len()-1);
                let mutate_weight: usize = rand::thread_rng().gen_range(0, indiv.neural_network.layers[mutate_layer][mutate_neuron].output_weights.len());
                indiv.neural_network.layers[mutate_layer][mutate_neuron].output_weights[mutate_weight] = rand::thread_rng().gen::<f64>();
            }
        }
    }
}

fn main() {

    let mut ga: GeneticAlgorithm = GeneticAlgorithm::new();

    let mut training_data: Vec<(Vec<u32>, u32)> = Vec::new();
    for _ in 0..1000 {
        let dists: Vec<u32> = vec![rand::thread_rng().gen_range(0, 100), rand::thread_rng().gen_range(1, 100)];
        let expected: u32 = dists[0].min(dists[1]);
        training_data.push((dists, expected));
    }

    let mut test_data: Vec<(Vec<u32>, u32)> = Vec::new();
    for _ in 0..1000 {
        let dists: Vec<u32> = vec![rand::thread_rng().gen_range(0, 100), rand::thread_rng().gen_range(1, 100)];
        let expected: u32 = dists[0].min(dists[1]);
        test_data.push((dists, expected));
    }

    for i in 0..100000 {

        println!("====== Generation {} ======", i);

        let elitism_rate: f64 = if i%2 == 0 { 0.10_f64 } else { 0.02_f64 };
        let mutation_rate: f64 = if i%2 == 0 { 0.10_f64 } else { 0.20_f64 };

        println!("Training!");
        ga.training(&training_data, elitism_rate, mutation_rate);
        println!("Training end!");

        println!("Selection!");
        let best_score: f64 = ga.selection();
        println!("Selection end!");
        println!("First pop best score: {}", best_score);
        if best_score == 100.0_f64 {
            ga.population[0].neural_network.print_network();
            break;
        }
        println!("Breeding!");
        ga.breeding();
        println!("Breeding end!");
        println!("Mutation!");
        ga.mutation();
        println!("Mutation end!");
    }

    for i in 0..10 {

        println!("====== Generation {} ======", i);

        let elitism_rate: f64 = 0.02_f64;
        let mutation_rate: f64 = 0.10_f64;

        println!("Final test!");
        ga.training(&test_data, elitism_rate, mutation_rate);
        println!("Final test end!");

        println!("Selection!");
        ga.selection();
        println!("Selection end!");
        println!("Breeding!");
        ga.breeding();
        println!("Breeding end!");
        println!("Mutation!");
        ga.mutation();
        println!("Mutation end!");
    }
    ga.population[0].neural_network.print_network();
}