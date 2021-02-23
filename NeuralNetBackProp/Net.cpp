#include "Net.h"
#include "Neuron.h"


Net::Net(unsigned int input_layer, std::vector<unsigned int> hidden_layer, unsigned int output_layer)
{
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