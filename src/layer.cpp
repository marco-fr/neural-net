#include "layer.hpp"

Layer::Layer(int n_inputs, int n_neurons, bool random_initialization, int prev_neurons){
    num_inputs = n_inputs;
    num_neurons = n_neurons;
    weights = new std::vector<std::vector<double>> (n_neurons, std::vector<double>(n_inputs, 0));
    bias = new std::vector<double>(n_neurons, 0.0);
    output = new std::vector<double>(n_neurons, 0.0);
    prev_neuron_deriv = new std::vector<double>(prev_neurons, 0.0);
    if(random_initialization){
        for(auto &i : *weights){
            for(auto &j : i){
                while(j == 0)
                    j = gen_random(-1.0, 1.0);
            }
        }
        for(auto &i : *bias){
            while(i == 0)
                i = gen_random(-1.0, 1.0);
        }
    }
}

double Layer::relu_activation(double val){
    return std::max(val, 0.0);
}

double Layer::linear_activation(double val){
    return val;
}

double Layer::relu_derivative(double val){
    if(val == 0) return 0;
    return 1;
}

double Layer::sigmoid_activation(double val){
    return 1.0/(1 + exp(-val));
}

double Layer::sigmoid_derivative(double val){
    return sigmoid_activation(val) * (1.0 - (sigmoid_activation(val)));
}

double Layer::leaky_relu_activation(double val){
    return std::max(val, 0.1* val);
}

double Layer::leaky_relu_derivative(double val){
    if(val <= 0) return 0.1;
    return 1;
}

void Layer::delete_resize_output(int size){
    if(output->size() == size) return;
    delete output;
    output = new std::vector<double>(size, 0.0);
}

double Layer::dot_product(std::vector<double> &a, std::vector<double> &b){
    double prod = 0;
    if(a.size() != b.size()) throw std::runtime_error("Dot product invalid size");
    for(int i = 0; i < a.size(); i++){
        prod += a[i] * b[i];
    }
    return prod;
}

std::vector<double>* Layer::compute_single_output(std::vector<double> &input, double (*activation_function)(double)){
    double temp;
    for(int i = 0; i < num_neurons; i++){
        temp = dot_product(input, (*weights)[i]);
        temp += (*bias)[i];
        (*output)[i] = activation_function(temp);
    }
    return output;
}

void Layer::clear_prev_deriv(){
    for(auto &i : *prev_neuron_deriv) i = 0;
}

double Layer::gen_random(double lower, double upper){
    return (rand() % 10000)/(10000.0 / (upper - lower))+ lower;
}

void Layer::print_weights(){
    std::cout << "Weights: " << std::endl;
    for(auto i : *weights){
        for(auto j : i){
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

void Layer::print_bias(){
    std::cout << "Bias: " << std::endl;
    for(auto i : *bias){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void Layer::print_output(){
    //std::cout << "Output Layer: " << std::endl;
    for(auto &i : *output){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void Layer::print_layer_info(){
    std::cout << "Layer Info - ";
    std::cout << "Inputs: " << num_inputs << " Neurons: " << num_neurons << std::endl;
}

Layer::~Layer(){
    delete weights;
    delete bias;
    delete output;
    delete prev_neuron_deriv;
}