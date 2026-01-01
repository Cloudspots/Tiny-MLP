#pragma once
#include <cmath>
#include "tools.h"
// https://www.luogu.com.cn/article/blfb9uj9
namespace activate_func
{
	// sigmoid
	template<typename _Valt>
	auto sigmoid(const _Valt& x) { return forall(x, [](const auto& xx) { return 1 / (1 + exp(-xx)); }); }
	// d(sigmoid)/dx
	template<typename _Valt>
	auto d_sigmoid(const _Valt& x) { return forall(x, [](const auto& xx) { auto k = sigmoid(xx); return k * (1 - k); }); }

	// HardSigmoid
	template<typename _Valt>
	auto hard_sigmoid(const _Valt& x) { return forall(x, [](const auto& xx) { return max(0, min(1, xx * (1.0L / 6) + 1.0L / 3)); }); }
	// d(HardSigmoid)/dx
	template<typename _Valt>
	auto d_hard_sigmoid(const _Valt& x) { return forall(x, [](const auto& xx) { return xx < -2 || xx > 4 ? 0 : (_Valt)1 / 6; }); }

	// ReLU
	template<typename _Valt>
	auto ReLU(const _Valt& x) { return forall(x, [](const auto& xx) { return xx > 0 ? xx : 0; }); }
	// d(ReLU)/dx
	template<typename _Valt>
	auto d_ReLU(const _Valt& x) { return forall(x, [](const auto& xx) { return xx > 0 ? 1 : 0; }); }

	// tanh
	template<typename _Valt>
	auto tanh(const _Valt& x) { return forall(x, [](const auto& xx) { return 2 / (1 + exp(-2 * xx)) - 1; }); }
	// d(tanh)/dx
	template<typename _Valt>
	auto d_tanh(const _Valt& x) { return forall(x, [](const auto& xx) { auto v = tanh(xx); return 1 - v * v; }); }

	// HardTanh
	template<typename _Valt>
	auto hard_tanh(const _Valt& x) { return forall(x, [](const auto& xx) { return max(-1, min(1, xx)); }); }
	// d(HardTanh)/dx
	template<typename _Valt>
	auto d_hard_tanh(const _Valt& x) { return forall(x, [](const auto& xx) { return xx < -1 || xx > 1 ? 0 : 1; }); }

	//   Leaky ReLU + PReLU
	// = Leaky PReLU
	template<typename _Valt, typename _At = _Valt>
	auto Leaky_PReLU(const _Valt& x, const _At& a) { return forall(x, [a](const auto& xx) { return xx > 0 ? xx : a * xx; }); }
	// d(Leaky PReLU)/dx
	template<typename _Valt, typename _At = _Valt>
	auto d_Leaky_PReLU(const _Valt& x, const _At& a) { return forall(x, [a](const auto& xx) { return xx > 0 ? 1 : a; }); }

	// ELU
	template<typename _Valt, typename _At = _Valt>
	auto ELU(const _Valt& x, const _At& a) { return forall(x, [a](const auto& xx) { return xx > 0 ? xx : a * (exp(xx) - 1); }); }
	// d(ELU)/dx
	template<typename _Valt, typename _At = _Valt>
	auto d_ELU(const _Valt& x, const _At& a) { return forall(x, [a](const auto& xx) { return xx > 0 ? 1 : a * exp(xx); }); }

	// Swish / SiLU
	template<typename _Valt>
	auto swish(const _Valt& x) { return forall(x, [](const auto& xx) { return xx * sigmoid(xx); }); }
	// d(Swish)/dx
	// sigmoid 怎么你了，为什么要这么玩 sigmoid.jpg
	template<typename _Valt>
	auto d_swish(const _Valt& x) { return forall(x, [](const auto& xx) { return sigmoid(xx) + xx * sigmoid(xx) * (1 - sigmoid(xx)); }); }

	// HardSwish
	template<typename _Valt>
	auto hard_swish(const _Valt& x) { return forall(x, [](const auto& xx) { return (xx <= -3 ? 0 : (xx >= 3 ? xx : xx * (xx + 3) / 6)); }); }
	// d(HardSwish)/dx
	template<typename _Valt>
	auto d_hard_swish(const _Valt& x) { return forall(x, [](const auto& xx) { return xx < -3 ? 0 : (xx > 3 ? 1 : xx / 3 + (_Valt)1 / 2); }); }

	// Softmax
	template<typename _Valt, typename _Tt>
	auto softmax(const _Valt& x, const _Tt temp = 1)
	{
		decltype(*x.begin()) sum = 0;
		for (const auto& y : x)
		{
			sum += exp(y / temp);
		}
		_Valt res = x;
		for (auto& y : res)
		{
			y = exp(y / temp) / sum;
		}
		return res;
	}
	// ∂(Softmax)/∂xx
	template<typename _Valt, typename _Kt, typename _Tt>
	auto d_softmax(const _Valt& x, const _Kt delta, const _Tt temp = 1)
	{
		decltype(*x.begin()) sum = 0;
		for (const auto& y : x)
		{
			sum += exp(y / temp);
		}
		return exp(delta) / sum;
	}

	// Softplus
	template<typename _Valt>
	auto softplus(const _Valt& x) { return forall(x, [](const auto& xx) { return log(1 + exp(xx)); }); }
	// d(Softplus)/dx
	template<typename _Valt>
	auto d_softplus(const _Valt& x) { return forall(x, [](const auto& xx) { if (xx > 0) return sigmoid(xx); auto k = exp(xx); return k / (1 + k); }); }

	// Mish
	// 怎么能乱堆叠呢
	template<typename _Valt>
	auto mish(const _Valt& x) { return forall(x, [](const auto& xx) { return xx * tanh(softplus(xx)); }); }
	// d(Mish)/dx
	template<typename _Valt>
	auto d_mish(const _Valt& x) { return forall(x, [](const auto& xx) { return d_tanh(softplus(xx)) * d_softplus(xx) + tanh(softplus(xx)); }); }

	// AconC/MetaAconC
	template<typename _Valt, typename _At = _Valt>
	auto AconC(const _Valt& x, const _At& a) { return forall(x, [&](const auto& xx) { return 1 / (exp(-1 * a * xx) + 1); }); };
};