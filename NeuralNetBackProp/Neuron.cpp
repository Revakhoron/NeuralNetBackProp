#include "Neuron.h"
#include "Connection.h"


Neuron::Neuron(double val, type e) : n_type(e)
{
	double value = 0.0;
	set_output_value(value);

}
void Neuron::set_output_value(double &value)
{
	this->output_val = value;
}

double Neuron::get_output_value() const
{
	return output_val;
}

void Neuron::weighted_sum()
{
	double sum = 0.0;
	for (auto& connection : input_connections)
	{
		sum += (connection->input_neuron->get_output_value() * connection->get_weight());
	}
	//std::cout << "weighted sum(), sum before transfer function" << sum << std::endl;
	set_input_val(sum);
    sum = transfer_function(sum);
	//std::cout << "weighted sum(), sum after transfer function" << sum << std::endl;
	this->set_output_value(sum);
}

void Neuron::add_input_connection(std::shared_ptr<Connection> conn)
{

	input_connections.push_back(conn);
}

void Neuron::add_output_connection(std::shared_ptr<Connection> conn)
{
	output_connections.push_back(std::move(conn));
}

double Neuron::transfer_function(double val)
{
	return std::tanh(val);
}

double Neuron::transfer_function_deriv(double val)
{
	return (1.0 - (std::tanh(val) * std::tanh(val)));
	//return 1.0 - std::tanh(val * val);
	//return 1.0 - (val * val);
}

double Neuron::get_gradient() const
{
	return gradient_val;
}

void Neuron::calc_gradient()
{
	//std::cout << "calc gradient";
	double var = 0.0;
	if ((n_type == type::hidden) || (n_type == type::input))
	{
		for (auto& o_conn : output_connections)
		{
			var += o_conn->get_weight() * o_conn->output_neuron->get_gradient();
			
		}

		gradient_val = transfer_function_deriv(input_val) * var;
	}
	else
	{
		std::cout << "this is an output neuron!";
		EXIT_FAILURE;
	}
}

void Neuron::calc_gradient(const double& target)
{
	if (n_type == type::output)
	{
		double var = (target-output_val);
		gradient_val =  var * transfer_function_deriv(input_val);
	}
	else
	{
		std::cout << "this is not an output neuron!";
		EXIT_FAILURE;
	}

}

void Neuron::update_input_weights()
{
	double old_value = 0.0;
	double new_value = 0.0;
	if (!input_connections.empty())
	{
		for (auto& input_weight : input_connections)
		{
			old_value = input_weight->get_delta_weight();

			//new_value = ((learning_rate * input_weight->input_neuron->output_val * gradient_val));
			new_value = ((learning_rate * input_weight->input_neuron->output_val * gradient_val));
			input_weight->delta_weight = new_value;
			input_weight->weight += new_value;
		}
	}
}
