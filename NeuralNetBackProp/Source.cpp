#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned>& topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double>& inputVals);
    unsigned getTargetOutputs(vector<double>& targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned>& topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
    targetOutputVals.clear();

    
    
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}



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
            o_vals += to_string(neuron.get_output_value()) + ",";
            i_vals += to_string(neuron.get_input_val()) + ",";
            g_vals += to_string(neuron.get_gradient_val()) + ",";
        }
    }
    o_file << "output_values: {" << o_vals << "}" << "\n";
    o_file << "input_values: {" << i_vals << "}" << "\n";
    o_file << "gradient_values: {" << g_vals << "}" << "\n";

}


int main()
{
    //TrainingData trainData("/tmp/trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    vector<unsigned> input;
    vector<unsigned> hidden;
    vector<unsigned> output;

    //trainData.getTopology(topology);
    vector<vector<double>> input_vals = { {0.0,0.0},{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0} };
    vector<double> output_val = { 0.0,1.0,0.0,1.0 };

   /* vector<double> input_vals = { 1.0,0.0 };
    vector<double> output_vals = { 0.0 };*/

    vector<double> resulterino;

    Net my_net(2, {2,2},1);
    int iters = 0;
    int i = 0;
    vector<double> current_output = { 0.0 };
    vector<double> current_input_vals = {0.0,0.0};
    vector<double> true_resulterino;
    while (iters < 20000)
    {
        if (i == 4)
        {
            i = 0;
        }
        for (int j = 0; j < 2; j++)
        {
            current_input_vals[j] = input_vals[i][j];
        }

        //std::cout << "current input vals: " << current_input_vals[0] << ", " << current_input_vals[1] << std::endl;
        current_output[0] = output_val[i];
        //std::cout << "current output vals: " << current_output[0] << std::endl;

        
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
    

    std::ofstream o_f("nn.txt");
    
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
    


   /* for (auto& a : resulterino)
    {
        std::cout << a << " ,";
    }*/



    cout << endl << "Done" << endl;
	return 0;
} 