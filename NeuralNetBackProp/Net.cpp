#include "Net.h"
#include "Neuron.h"
#include <fstream>


int Net::values_from_string(std::string&& search, std::string& file)
{
	int index = 0;
	std::string zkouska = "";
	index = file.find(search);

	for (auto it = file.begin() + index + search.size(); it != file.cend(); ++it)
	{
		if (*it == '\n')
			break;
		zkouska += *it;
	}
	return ((file[index + search.size()]) - '0');
}

template<typename T>
void Net::values_from_string(std::string&& search, std::string& file, std::vector<T>& values)
{
	std::string var;

	int index = 0;
	std::string zkouska = "";
	index = file.find(search);

	if (std::is_same<T, double>::value)
	{
		double val = 0.0;
		std::string temp = "";
		for (auto it = file.begin() + index + search.size(); it != file.cend(); ++it)
		{
			if (*it == '\n')
				break;
			if (*it == ',' || *it == '}' || (*it == '{') || ((*it == ' ')))
			{
				if (!var.empty())
				{
					val = std::stod(var);
					var = "";
					values.emplace_back(val);
				}
				std::cout << *it;

			}
			else
			{
				var += *it;
			}
		}
	}
	if (std::is_same<T, int>::value)
	{
		int val = 0;
		std::cout << "file reading:" << std::endl;
		for (auto it = file.begin() + index + search.size(); it != file.cend(); ++it)
		{

			//std::cout << *it;
			if (*it == '\n')
				break;
			if ((*it == ',') || (*it == '{') || (*it == '}'))
			{
				continue;
			}
			else
			{
				val = *it - '0';
				values.emplace_back(val);
			}

		}
	}
}

Net::Net(unsigned int input_layer, std::vector<unsigned int> hidden_layer, unsigned int output_layer)
{
	if ((input_layer <= 0) || (hidden_layer.empty()) || (output_layer <= 0))
	{
		std::cout << "Cannot construct a neural network from this parameters!";
		EXIT_FAILURE;
	}

	Layer i_layer;
	for (int i = 0; i < input_layer; i++)
	{
		i_layer.push_back(Neuron(0.0,Neuron::type::input));
	}
	net.push_back(i_layer);
	Layer h_layer;
	for (auto& a : hidden_layer)
	{
		for (int i = 0; i < a; i++)
		{
			h_layer.push_back(Neuron(0.0, Neuron::type::hidden));
		}
		net.push_back(h_layer);
		h_layer.clear();
	}
	Layer o_layer;
	for (int i = 0; i < output_layer; i++)
	{
		o_layer.push_back(Neuron(0.0, Neuron::type::output));
	}
	net.push_back(o_layer);
	set_connections();
}

Net::Net(std::string& filename)
{
	std::ifstream nn_file(filename);
	std::string line;
	std::string file;

	int i_n = 0;
	std::vector<unsigned int> h_l;
	int o_l = 0;

	double l_r = 0.0;

	std::vector<double> weights;
	std::vector<double> output_vals;
	std::vector<double> input_vals;
	std::vector<double> gradient_vals;

	if (nn_file.is_open())
	{
		while (std::getline(nn_file, line)) 
		{
			file += line + "\n";
		}

		if (file.empty())
		{
			std::cout << "bad read from a file";
			EXIT_FAILURE;
		}
		i_n = values_from_string("input_neurons: ", file);
		values_from_string("hidden_l: ", file, h_l);
		o_l = values_from_string("output_neurons: ", file);
		values_from_string("weights: ", file, weights);
		values_from_string("output_values: ", file, output_vals);
		values_from_string("input_values: ", file, input_vals);
		values_from_string("gradient_values: ", file, gradient_vals);

		Net(i_n, h_l, o_l);

		int iter = 0;
		int n_iter = 0;

		for (auto& l : net)
		{
			for (auto& n : l)
			{
				n.set_output_value(output_vals[n_iter]);
				n.set_input_val(input_vals[n_iter]);
				n.set_gradient_val(gradient_vals[n_iter]);
				for (auto& conn : n.output_connections)
				{
					conn->set_weight(weights[iter]);
					iter++;
				}
			}
		}
		iter = 0;
		n_iter = 0;
	}
	else
	{
		std::cout << "could not open a file";
		EXIT_FAILURE;
	}

}


//void Net::set_connections()
//{
//	for (int i = 0; i < net.size()-1; i++)
//	{
//		for (auto& in : net[i])
//		{
//			for (auto& out : net[i + 1])
//			{
//				in.add_output_connection(Connection(&in, &out));
//				out.add_input_connection(Connection(&in, &out));
//			}
//		}
//	}
//}

void Net::set_connections()
{
	std::vector<Connection> conns{};
	for (int i = 0; i < net.size() - 1; i++)
	{
		connections.push_back(conns);
	}

	for (int i = 0; i < net.size() - 1; i++)
	{
		for (auto& in : net[i])
		{
			for (auto& out : net[i + 1])
			{
				Connection c(&in, &out);
				connections[i].push_back(c);
				in.add_output_connection(std::make_shared<Connection>(connections[i].back()));
				out.add_input_connection(std::shared_ptr<Connection>(in.output_connections.back()));

				//in.add_output_connection(std::make_shared<Connection>(Connection(&in, &out)));
				//out.add_input_connection(std::make_shared<Connection>(Connection(&in, &out)));
			}
		}
	}
}

void Net::back_propagation(std::vector<double>& targets)
{
	unsigned int i_output = net.size()-1;
	//Layer* output_layer = &net.back();
	unsigned int o_size = net.back().size();
	error = 0.0;
	double delta_val = 0.0;
	for (int i = 0; i < o_size; i++)
	{
		delta_val = targets[i] - net[i_output][i].get_output_value();
		error += std::pow(delta_val, 2);
	}
	error /= o_size;
	error = std::sqrt(error);

	for (int i = 0; i < o_size; i++)
	{
		net[i_output][i].calc_gradient(targets[i]);
	}

	for (int i = net.size()-2; i >= 0; i--)
	{
		for (auto& neuron : net[i])
		{
			neuron.calc_gradient();
		}
	}

	for (int i = net.size() - 1; i > 0; i--)
	{
		for (auto& neuron : net[i])
		{
			neuron.update_input_weights();
		}
	}



}

void Net::feed_forward(std::vector<double>& input)
{
	for (int i = 0; i < input.size(); i++)
	{
		net[0][i].set_output_value(input[i]);
	}

	for (int j = 1; j < net.size(); j++)
	{
		for (int n = 0; n < net[j].size(); n++)
		{
			net[j][n].weighted_sum();
		}
	}
}

void Net::get_results(std::vector<double>& results)
{
	//results.clear();
	for (auto& output_neruons : net.back())
	{
		results.push_back(output_neruons.get_output_value());
	}
}