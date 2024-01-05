#include "network.hpp"

NNetwork::NNetwork(int n_inputs, int n_outputs, int layers, int neurons_per_layer){
    srand((unsigned int)time(NULL));
    num_inputs = n_inputs, num_outputs = n_outputs;
    num_layers = layers, num_neurons = neurons_per_layer;
    int prev_neurons;
    hidden_layers = new Layer*[layers];
    int temp_inputs, temp_outputs; 
    for(int i = 0; i < layers; i++){
        temp_inputs = temp_outputs = neurons_per_layer;
        if(i == 0) temp_inputs = n_inputs, prev_neurons = n_inputs;
        if(i == layers - 1) temp_outputs = n_outputs;
        hidden_layers[i] = new Layer(temp_inputs, temp_outputs, true, prev_neurons);
        prev_neurons = temp_outputs;
    }
}

std::vector<double>* NNetwork::compute_network_single_input(std::vector<double> &input){
    std::vector<double> *temp_input = &input;
    for(int i = 0; i < num_layers - 1; i++){
        temp_input = hidden_layers[i]->compute_single_output(*temp_input, Layer::relu_activation);
    }
    temp_input = hidden_layers[num_layers - 1]->compute_single_output(*temp_input, Layer::linear_activation);
    return temp_input;
}

double NNetwork::compute_cost(double estimate, double correct){
    return 0.5 * (estimate - correct) * (estimate - correct);
}

double NNetwork::cost_derivative(double estimate, double correct){
    return (estimate - correct);
}

double NNetwork::compute_total_cost(std::vector<double> &estimate, std::vector<double> &correct){
    double cost = 0;
    if(estimate.size() != correct.size()){
        throw std::runtime_error("Mismatched Cost Size");
    }
    for(int i = 0; i < estimate.size(); i++){
        cost += compute_cost(estimate[i], correct[i]);
    }
    return cost;
}

void NNetwork::backpropagation(std::vector<double> &input, std::vector<double> &correct){
    compute_network_single_input(input);
    Layer *cur;
    double change;
    double multiplier = 1.0/1000.0;
    for(int cur_layer = num_layers - 1; cur_layer >= 0; cur_layer--){
        cur = hidden_layers[cur_layer];
        cur->clear_prev_deriv();
        for(int i = 0; i < cur->num_neurons; i++){
        // Last layer of weights
            for(int j = 0; j < cur->num_inputs; j++){
                change = Layer::relu_derivative((*(cur->output))[i]);
                if(cur_layer == num_layers - 1){
                    change *= cost_derivative((*(cur->output))[i], correct[i]);
                }
                else{
                    change *= (*(hidden_layers[cur_layer + 1]->prev_neuron_deriv))[j];
                }
                // Prev Neuron Deriv
                (*(cur->prev_neuron_deriv))[j] += change * (*(cur->weights))[i][j];
                change *= multiplier;
                (*(cur->bias))[i] -= change;
                if(cur_layer == 0){
                    change *= input[j]; // Weight Derivative;
                }
                else{
                    change *= (*(hidden_layers[cur_layer - 1]->output))[j]; // Weight Derivative;
                }
                (*(cur->weights))[i][j] -= change;
            }
        }
        for(int j = 0; j < cur->num_inputs; j++){
            if(cur_layer > 0){
               (*(cur->prev_neuron_deriv))[j] /= (cur->num_neurons + 0.0);
            }
        }

    }
}

void NNetwork::print_network_info(){
    for(int i = 0; i < num_layers; i++){
        hidden_layers[i]->print_layer_info();
    }
}

void NNetwork::print_full_network(){
    for(int i = 0; i < num_layers; i++){
        hidden_layers[i]->print_layer_info();
        hidden_layers[i]->print_weights();
        hidden_layers[i]->print_bias();
        hidden_layers[i]->print_output();
    }
}

void NNetwork::print_output(){
    hidden_layers[num_layers - 1]->print_output();
}


NNetwork::~NNetwork(){
    for(int i = 0; i < num_layers; i++){
        delete hidden_layers[i];
    }
    delete[] hidden_layers;
}