#pragma once
#include <iostream>
#include <random>

//funkce pro generaci nahodnych cisel v rozmezi a-b
inline float rng(const float& a, const float& b)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist6(a, b);

    return (dist6(rng));
}