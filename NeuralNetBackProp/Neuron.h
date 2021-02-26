#pragma once
#include <iostream>
#include <vector>
#include "Connection.h"

class Neuron
{
	friend class Connection;
public:
	enum class type
	{
		input,
		hidden,
		output
	} n_type;
	Neuron(double,type e);
	~Neuron() = default;
	void set_output_value(double&);
	double get_output_value() const;
	void weighted_sum();
	void add_input_connection(std::shared_ptr<Connection>);
	void add_output_connection(std::shared_ptr<Connection>);
	double transfer_function(double);

	double transfer_function_deriv(double);
	void update_input_values(std::vector<Connection>&);

	double get_gradient() const;
	double get_input_val() const{ return input_val; };
	double get_gradient_val() const { return gradient_val; };
	void set_gradient_val(double val) { gradient_val = val; };
	void set_input_val(double val) { input_val = val; };
	void calc_gradient();
	void calc_gradient(const double&);
	void update_input_weights();
	std::vector<std::shared_ptr<Connection>> input_connections;
	std::vector<std::shared_ptr<Connection>> output_connections;

	static constexpr double learning_rate = 0.15;
private:
	double output_val;
	double input_val;
	double gradient_val;
};
