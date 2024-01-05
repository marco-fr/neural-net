#include "network.hpp"
#include <cmath>
#include <iostream>
#include <random>

int main(){
    NNetwork *t = new NNetwork(1, 1, 3, 16);
    //std::vector<double> b = {1, 1, 1};
    //std::vector<double> a = {1, 5, 1, 1};
    //std::vector<double> *ans = t->compute_network_single_input(b);
    //t->compute_network_single_input(b);
    //t->print_full_network();
    //for(int i = 0; i < 1000; i++){
    //t->backpropagation(b, a);
    //}
    //t->compute_network_single_input(b);
    //t->print_full_network();
    double x = 0.0;
    std::vector<double> in = {3.14};
    t->compute_network_single_input(in);
    t->print_full_network();
    /*
    for(int j = 0; j < 10; j++){
    for(int i = 0; i < 100; i++){
        for(x = (6.28/10.0)*j; x <= (6.28/10.0)*(j+1); x += (6.28/ 1000.0)){
            std::vector<double> in = {x};
            std::vector<double> out = {sin(x) + 2.0};
            //std::vector<double> out = {0.2*x + 0.5};
            t->backpropagation(in, out);
        }
    }
    }*/
    std::uniform_real_distribution<double> unif(0.0, 6.28);
    std::default_random_engine re;

    for(int i = 0; i < 20000; i++){
        /*for(int j = 0; j < 100; j++){
            std::vector<double> in = {unif(re)};
            std::vector<double> out = {sin(in[0]) + 2.0};
            t->backpropagation(in, out);
        }*/
        for(x = 0.0; x <= 6.28; x += (6.28/ 100.0)){
            std::vector<double> in = {x};
            std::vector<double> out = {sin(x) + 2.0};
            //std::vector<double> out = {5.0*x + 5.0};
            t->backpropagation(in, out);
        }
    }
    for(x = 0.0; x <= 6.28; x += (6.28/ 100.0)){
        std::vector<double> in = {x};
        t->compute_network_single_input(in);
        std::cout << x << ", ";
        t->print_output();
    }
    //std::cout << sin(3.14) + 2 << std::endl;
    t->print_full_network();
    delete t;
    return 0;
}