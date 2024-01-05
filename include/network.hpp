#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <math.h>
#include "layer.hpp"

class NNetwork{
    public:
    NNetwork(int n_inputs, int n_outputs, int layers, int neurons_per_layer);
    std::vector<double>* compute_network_single_input(std::vector<double> &input);
    double compute_cost(double estimate, double correct);
    double cost_derivative(double estimate, double correct);
    double compute_total_cost(std::vector<double> &estimate, std::vector<double> &correct);
    void backpropagation(std::vector<double> &input, std::vector<double> &correct);
    void print_network_info();
    void print_full_network();
    void print_output();
    ~NNetwork();

private:
    int num_inputs, num_neurons, num_layers, num_outputs;
    Layer **hidden_layers;
};

#endif