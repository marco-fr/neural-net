#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>

class Layer{
    public:
    Layer(int n_inputs, int n_neurons, bool random_initialization, int prev_nuerons);

    ~Layer();

    void print_weights();
    void print_bias();
    void print_output();
    void print_layer_info();
    std::vector<double>* compute_single_output(std::vector<double> &input, double (*activation_function)(double));
    void clear_prev_deriv();
    double static relu_activation(double val);

    double static linear_activation(double val);

    double static relu_derivative(double val);
    double static sigmoid_activation(double val);
    double static sigmoid_derivative(double val);
    double static leaky_relu_activation(double val);
    double static leaky_relu_derivative(double val);
    std::vector<std::vector<double>> *weights;
    std::vector<double> *bias, *output;
    std::vector<double> *prev_neuron_deriv;

    int num_neurons, num_inputs;

    //std::vector<std::vector<double>> *weights, *inputs, *output;
    private:

    double gen_random(double lower, double upper);
    void delete_resize_output(int size);
    double dot_product(std::vector<double> &a, std::vector<double> &b);
};

#endif