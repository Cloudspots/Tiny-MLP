#pragma once
#include <vector>
#include <random>
#include "tools.h"
#include "matrix.h"

namespace init_func
{
	// Xavier/Glorot Uniform Distribution
	template<typename T>
	T Xavier_uniform(size_t n_in, size_t n_out)
	{
		static std::random_device rd;
		static std::mt19937_64 mt(rd() + (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count());
		T w = sqrt(T(6) / (n_in + n_out));
		std::uniform_real_distribution<T> u(-w, w);
		return u(mt);
	}
	// Xavier/Glorot Gauss Distribution
	template<typename T>
	T Xavier_gauss_once(size_t n_in, size_t n_out)
	{
		static std::random_device rd;
		static std::mt19937_64 mt(rd() + (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count());
		T w = sqrt(T(2) / (n_in + n_out));
		std::normal_distribution<T> u(0, w);
		return u(mt);
	}
	template<typename T>
	std::vector<type_matrix<T>> Xavier_gauss(const std::valarray<size_t>& sz)
	{
		std::vector<type_matrix<T>> res;
		res.push_back({});
		for (unsigned i = 1; i < sz.size(); i++)
		{
			res.push_back(type_matrix<T>(sz[i], sz[i - 1]));
			for (auto& p : res.back())
			{
				p = Xavier_gauss_once<T>(sz[i - 1], sz[i]);
			}
		}
		return res;
	}
	// He
	template<typename T>
	T He_uniform(size_t n_in) { return Xavier_uniform<T>(n_in, 0); }
	template<typename T>
	T He_gauss(size_t n_in) { return Xavier_guass<T>(n_in, 0); }
	template<typename T>
	T bias_init_once(T q = T()) { return q; }
	template<typename T>
	std::vector<type_matrix<T>> bias_init(const std::valarray<size_t>& sz, T q = T())
	{
		std::vector<type_matrix<T>> res;
		for (unsigned i = 0; i < sz.size(); i++)
		{
			res.push_back(type_matrix<T>(sz[i], 1));
			for (unsigned j = 0; j < sz[i]; j++)
			{
				for (auto& k : res.back())
				{
					k = bias_init_once(q);
				}
			}
		}
		return res;
	}
}