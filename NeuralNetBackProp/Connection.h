#pragma once
class Neuron;

class Connection
{
	friend class Neuron;
public:
	Connection(Neuron* input, Neuron* output);
	void set_weight(double);
	void set_delta_weight(double);
	double get_weight() const;
	double get_delta_weight() const;
private:
	double weight;
	double delta_weight;
	Neuron* input_neuron;
	Neuron* output_neuron;
};