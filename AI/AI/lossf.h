#pragma once
#include <cmath>
#include <vector>
#include <valarray>
#include <stdexcept>
#include "tools.h"

namespace loss_func
{
	// Binary Cross-Entropy
	template<typename T>
	T BCE(const T& model_output, const T& real_output, const T& eps = 1e-7)
	{
		if (model_output < eps || 1 - model_output < eps) return MSE(model_output, real_output); // log(model_output) is dangerous. Don't do that. Use MSE for safety. 
		return -(real_output * log(model_output) + (1 - real_output) * log(1 - model_output));
	}
	// d(BCE)/dp
	template<typename T>
	T d_BCE(const T& model_output, const T& real_output, const T& eps = 1e-7)
	{
		if (model_output < eps || 1 - model_output < eps) return d_MSE(model_output, real_output);
		// d(aln(b)) / db = a/b
		// d((1-a)ln(1-b)) / db = (1-a)/(b-1)
		return -(real_output / model_output + (1 - real_output) / (model_output - 1));
	}

	// Categorical Cross-Entropy
	template<typename T>
	T CCE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in CCE: The length of p(model's output) and y(correct output) should be the same.");
		T sum{};
		for (unsigned i = 0; i < p.size(); i++)
		{
			sum += -(y[i] * log(p[i]));
		}
		return sum;
	}
	template<typename T>
	std::valarray<T> d_CCE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in d_CCE: The length of p(model's output) and y(correct output) should be the same.");
		return -(y / p);
	}
	template<typename T>
	T g_CCE(const std::valarray<T>& p, const std::valarray<T>& y, size_t id)
	{
		if (p.size() != y.size()) throw std::length_error("Error in d_CCE: The length of p(model's output) and y(correct output) should be the same.");
		if (id >= p.size()) throw std::out_of_range("Error in d_CCE: id should less than the size of p.");
		return -(y[id] / p[id]);
	}
	// Mean Squared Error
	template<typename T>
	T MSE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in MSE: The length of p(model's output) and y(correct output) should be the same.");
		T sum{};
		for (unsigned i = 0; i < p.size(); i++)
		{
			T r = p[i] - y[i];
			sum += r * r;
		}
		return sum / p.size();
	}
	// Mean Squared Error
	template<typename T>
	std::valarray<T> d_MSE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in MSE: The length of p(model's output) and y(correct output) should be the same.");
		return 2 * (p - y) / p.size();
	}
	// Mean Absolute Error
	template<typename T>
	T MAE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in MSE: The length of p(model's output) and y(correct output) should be the same.");
		T sum{};
		for (int i = 0; i < p.size(); i++)
		{
			sum += abs(p[i] - y[i]);
		}
		return sum / p.size(); 
	}
	template<typename T>
	std::valarray<T> d_MAE(const std::valarray<T>& p, const std::valarray<T>& y)
	{
		if (p.size() != y.size()) throw std::length_error("Error in MSE: The length of p(model's output) and y(correct output) should be the same.");
		std::valarray<T> vr;
		vr.resize(p.size());
		for (int i = 0; i < p.size(); i++)
		{
			vr[i] = p[i] > y[i] ? 1 : -1;
		}
		return vr / p.size();
	}
	// ___     ___
	// |H|uber |L|oss
	// |H|ello |L|ibreOJ(
	// ^^^     ^^^
	template<typename T>
	T Huber(const T& p, const T& y, const T& delta)
	{
		auto MAE_res = MAE(p, y);
		if (MAE_res <= delta) return MSE(p, y) / 2;
		else return delta * MAE_res - delta * delta / 2;
	}
	template<typename T>
	T d_Huber(const T& p, const T& y, const T& delta)
	{
		auto MAE_res = MAE(p, y);
		if (MAE_res <= delta) return y - p;
		else return delta * d_MAE(p, y);
	}
}