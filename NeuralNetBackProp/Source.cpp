#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>



void nn_file(const Net& nn, std::ofstream& o_file)
{
    o_file << "input_neurons: " << nn.net[0].size() << "\n";
    o_file << "hidden_l: " << "{";
    for (int i = 1; i < nn.net.size() - 1; i++)
    {
        if (i == 1)
        {
            o_file << nn.net[i].size();
        }
        else
        {
            o_file << "," << nn.net[i].size();
        }
        
    }
    o_file << "}" << "\n";

    o_file << "output_neurons: " << nn.net[nn.net.size() - 1].size() << "\n";
    o_file << "learning_rate: " << nn.net[0][0].learning_rate << "\n";
    o_file << "weights: " << "{";

    for (int i = 0; i < nn.net.size() - 1; i++)
    {
        for (auto& neuron : nn.net[i])
        {
            for (auto& conn : neuron.output_connections)
            {
                o_file << conn->get_weight() << ",";
            }
        }
    }
    o_file << "}" << '\n';

    std::string o_vals = "";
    std::string i_vals = "";
    std::string g_vals = "";

    for (auto& layer : nn.net)
    {
        for (auto& neuron : layer)
        {
            o_vals += std::to_string(neuron.get_output_value()) + ",";
            i_vals += std::to_string(neuron.get_input_val()) + ",";
            g_vals += std::to_string(neuron.get_gradient_val()) + ",";
        }
    }
    o_file << "output_values: {" << o_vals << "}" << "\n";
    o_file << "input_values: {" << i_vals << "}" << "\n";
    o_file << "gradient_values: {" << g_vals << "}" << "\n";

}


int main()
{
    std::vector<unsigned> topology;
    std::vector<unsigned> input;
    std::vector<unsigned> hidden;
    std::vector<unsigned> output;

    std::vector<std::vector<double>> input_vals = { {0.0,0.0},{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0} };
    std::vector<double> output_val = { 0.0,1.0,0.0,1.0 };
    std::vector<double> resulterino;

    Net my_net(2, {16},1);
    //Net my_net("nn.txt");
    int iters = 0;
    int i = 0;
    std::vector<double> current_output = { 0.0 };
    std::vector<double> current_input_vals = {0.0,0.0};
    std::vector<double> true_resulterino;
    while (iters < 30000)
    {
        if (i == 4)
        {
            i = 0;
        }
        for (int j = 0; j < 2; j++)
        {
            current_input_vals[j] = input_vals[i][j];
        }
        current_output[0] = output_val[i];
        true_resulterino.push_back(output_val[i]);

        /*std::cout << "input neurons o_values" << std::endl;
        for (auto &a : my_net.net[0])
        {
            std::cout << a.get_output_value() << " ,";
        }
        std::cout << std::endl;
        std::cout << "hidden neurons o_values" << std::endl;
        for (auto& a : my_net.net[1])
        {
            std::cout << a.get_output_value() << " ,";
        }
        std::cout << "output neurons o_values" << std::endl;
        for (auto& a : my_net.net[2])
        {
            std::cout << a.get_output_value() << " ,";
        }
        std::cout << std::endl;*/

        my_net.feed_forward(current_input_vals);
        my_net.get_results(resulterino);
        my_net.back_propagation(current_output);

       /* std::cout << "input connections: " << std::endl;
        std::cout << my_net.net[1][0].input_connections[0]->get_weight() << std::endl;
        std::cout << my_net.net[1][0].input_connections[1]->get_weight() << std::endl;
        std::cout << "output connections: " << std::endl;
        std::cout << my_net.net[1][0].output_connections[0]->get_weight() << std::endl;*/
      

        //my_net.feed_forward(input_vals);
        //my_net.get_results(resulterino);
        //my_net.back_propagation(output_vals);

        iters++;
        i++;
    }
    

    for (int a = 0; a < true_resulterino.size(); a++)
    {
        std::cout << "true_resulterino" << true_resulterino[a] << ", nn result: " << resulterino[a];
        std::cout << std::endl;
    }
    std::cout << "before:" << std::endl;
    for (auto& l : my_net.net)
    {
        for (auto& n : l)
        {
            for (auto& c : n.output_connections)
            {
                std::cout << c->get_weight() << " ,";
            }
            std::cout << std::endl;
        }
    }

   // my_net.update_values("nn.txt");

    std::cout << "after:" << std::endl;
    for (auto& l : my_net.net)
    {
        for (auto &n : l)
        {
            for (auto& c : n.output_connections)
            {
                std::cout << c->get_weight() << " ,";
            }
            std::cout << std::endl;
        }
    }


    /*std::ofstream o_f("nn.txt");
    
    if (!o_f)
    {
        std::cout << "error opening file";
    }
    else
    {
        std::cout << "file opened";
    }

    o_f.clear();

    nn_file(my_net, o_f);

    o_f.close();
    */


   /* for (auto& a : resulterino)
    {
        std::cout << a << " ,";
    }*/



    std::cout << std::endl << "Done" << endl;
	return 0;
} 