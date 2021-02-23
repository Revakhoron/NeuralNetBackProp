#include "Connection.h"
#include "Neuron.h"
#include "rng.h"

Connection::Connection(Neuron* input, Neuron* output) : input_neuron(input), output_neuron(output)
{
	set_weight(rng(-0.3,0.3));
	delta_weight = 0.0;
}


void Connection::set_weight(double value)
{
	weight = value;
}


double Connection::get_weight() const
{
	return weight;
}

double Connection::get_delta_weight() const
{
	return delta_weight;
}
