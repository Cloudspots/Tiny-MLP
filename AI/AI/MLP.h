#pragma once
// Lionblaze's friends
#include "allheaders.h"

// 多层感知器 MLP
template<typename _Value = long double>
class MLP
{
protected:
	using mxtype = type_matrix<_Value>;
	using vmxtype = std::vector<mxtype>;
	using sztype = size_t;
	using vsztype = std::valarray<sztype>;
private:
	vmxtype weight, bias;
	vsztype size; // 神经元层大小（包括输入层、输出层和隐藏层）
	std::function<mxtype(const mxtype&)> activatef; // 激活函数
	std::function<mxtype(const mxtype&)> dactivatef; // 激活函数的偏导
	std::function<vmxtype(const vsztype&)> winitf; // 权重初始化函数
	std::function<vmxtype(const vsztype&)> binitf; // 偏置初始化函数
	std::function<_Value(const mxtype&, const mxtype&)> lossf; // 损失函数/代价函数
	std::function<std::valarray<_Value>(const mxtype&, const mxtype&)> dlossf; // 损失函数的导数/偏导
public:
	MLP(const vsztype& sz, // 大小
		decltype(activatef) acf = [](const mxtype& in) { return activate_func::Leaky_PReLU(in, 0.01); },
		decltype(dactivatef) dacf = [](const mxtype& in) { return activate_func::d_Leaky_PReLU(in, 0.01); },
		decltype(winitf) winf = [](const vsztype& r) { return init_func::Xavier_gauss<_Value>(r); },
		decltype(binitf) binf = [](const vsztype& r) { return init_func::bias_init<_Value>(r); },
		decltype(lossf) losf = [](const mxtype& x, const mxtype& y) { return loss_func::MSE<_Value>(x, y); },
		decltype(dlossf) dlosf = [](const mxtype& x, const mxtype& y) { return loss_func::d_MSE<_Value>(x, y); }
	)
	{
		activatef = acf;
		dactivatef = dacf;
		winitf = winf;
		binitf = binf;
		size = sz;
		lossf = losf;
		dlossf = dlosf;
		if (sz.size() < 2) throw std::length_error("Error in MLP::MLP: The MLP should contain at least two layers(the input layer and the output layer), but the length of the size vector is " + std::to_string(sz.size()) + ".");
		weight = winitf(sz);
		bias = binitf(sz);
	}
	virtual vmxtype get(const mxtype& in)
	{
		// Argument Check
		if (in.size().second != 1) throw std::invalid_argument("Error in MLP::get: The column number of the input matrix should be 1 (The matrix should be a column vector).");
		if (in.size().first != size[0]) throw std::invalid_argument("Error in MLP::get: The rows of the input matrix should equals to the input layer.");
		// Calculate a
		vmxtype a;
		a.push_back(in);
		for (unsigned i = 1; i < size.size(); i++)
		{
			a.push_back(activatef(weight[i] * a.back() + bias[i]));
		}
		return a;
	}
	/// <summary>
	/// 获取单次训练结果，反向传播算法 Backpropagation BP
	/// </summary>
	/// <param name="beta">学习率</param>
	/// <param name="in">输入</param>
	/// <param name="out">正确输出</param>
	virtual std::tuple<_Value, decltype(weight), decltype(bias), vmxtype> train(const type_matrix<_Value>& in, const type_matrix<_Value>& out)
	{
		// Argument Check
		if (in.size().second != 1) throw std::invalid_argument("Error in MLP::train: The column number of the input matrix should be 1 (The matrix should be a column vector).");
		if (in.size().first != size[0]) throw std::invalid_argument("Error in MLP::train: The rows of the input matrix should equals to the input layer.");
		// Calculate a and z
		// (We cannot use get function here because we should get a and z but only a).
		vmxtype a, z;
		a.push_back(in);
		z.push_back(in);
		for (unsigned i = 1; i < size.size(); i++)
		{
			z.push_back(weight[i] * a.back() + bias[i]);
			a.push_back(activatef(z.back()));
		}
		_Value loss = lossf(a.back(), out);
		// æ
		decltype(a) dw, db, da, ae;
		dw.resize(a.size());
		da.resize(a.size());
		db.resize(a.size());
		ae.resize(a.size());
		da.back() = dlossf(a.back(), out);
		for (unsigned i = a.size() - 1; i >= 1; i--)
		{
			if (i + 1 != a.size()) da[i] = rotate(weight[i + 1]) * ae[i + 1];
			ae[i] = dot_p(da[i], dactivatef(z[i]));
			dw[i] = ae[i] * rotate(a[i - 1]);
			db[i] = da[i];
		}
		return { loss, dw, db, da };
	}
	/// <summary>
	/// 应用训练结果
	/// </summary>
	/// <param name="beta">学习率</param>
	/// <param name="dw">delta w</param>
	/// <param name="db">delta b</param>
	virtual void apply_train(const _Value& beta, const decltype(weight)& dw, const decltype(bias)& db)
	{
		for (unsigned i = 1; i < size.size(); i++)
		{
			weight[i] -= beta * dw[i];
			bias[i] -= beta * db[i];
		}
	}
	virtual _Value train_and_apply(const _Value& beta, const type_matrix<_Value>& in, const type_matrix<_Value>& out)
	{
		auto [loss, dw, db, da] = train(in, out);
		apply_train(beta, dw, db);
		return loss;
	}
	virtual void set_losf(
		decltype(lossf) losf = [](const mxtype& x, const mxtype& y) { return loss_func::MAE<_Value>(x, y); },
		decltype(dlossf) dlosf = [](const mxtype& x, const mxtype& y) { return loss_func::d_MAE<_Value>(x, y); })
	{
		lossf = losf;
		dlossf = dlosf;
	}
	virtual void set_acf(
		decltype(activatef) acf = [](const mxtype& in) { return activate_func::Leaky_PReLU(in, 0.01); },
		decltype(dactivatef) dacf = [](const mxtype& in) { return activate_func::d_Leaky_PReLU(in, 0.01); })
	{
		activatef = acf;
		dactivatef = dacf;
	}
	virtual ~MLP() {};
};