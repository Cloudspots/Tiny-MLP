#pragma once
#include <initializer_list>
#include <stdexcept>
#include <valarray>
#include <utility>
#include <vector>

// 矩阵类
template<typename _Valt>
class type_matrix
{
private:
	size_t n, m;
	std::valarray<std::valarray<_Valt>> _Val;
	class inner_vector
	{
		size_t k;
		type_matrix* tm;
		const type_matrix* tk;
	public:
		inner_vector() = delete;
		inner_vector(size_t x, type_matrix* t) : k(x), tm(t), tk(nullptr) {};
		inner_vector(size_t x, const type_matrix* t) : k(x), tm(nullptr), tk(t) {};
		inner_vector(const inner_vector& x) : k(x.k), tm(x.tm), tk(x.tk) {};
		_Valt& operator[](size_t y) { return tm->_Val[k][y]; }
		const _Valt& operator[](size_t y) const { return tk->_Val[k][y]; }
	};
protected: // 我证明一下我是会 protected 的
	class tmatrix_iterator
	{
	private:
		type_matrix* tm;
		const type_matrix* tk;
		size_t x, y;
	protected:
		size_t getx() const { return x; }
		size_t gety() const { return y; }
		std::pair<size_t, size_t> getxy() const { return { x, y }; } // ? ? ?
	public:
		tmatrix_iterator() = delete;
		tmatrix_iterator(type_matrix* t, size_t xx = 0, size_t yy = 0) : tm(t), tk(nullptr), x(xx), y(yy) {}
		tmatrix_iterator(const type_matrix* t, size_t xx = 0, size_t yy = 0) : tm(nullptr), tk(t), x(xx), y(yy) {}
		tmatrix_iterator(const tmatrix_iterator& yy) : tm(yy.tm), tk(yy.tk), x(yy.x), y(yy.y) {}

		friend bool operator==(const tmatrix_iterator& x, const tmatrix_iterator& y) { return x.tm == y.tm && x.tk == y.tk && x.x == y.x && x.y == y.y; }
		friend bool operator!=(const tmatrix_iterator& x, const tmatrix_iterator& y) { return !(x == y); }

		// 前自增 ++x operator++()
		tmatrix_iterator& operator++() { x += (y == (tm == nullptr ? tk->m : tm->m) - 1); y = (y + 1) % (tm == nullptr ? tk->m : tm->m); return *this; }
		// 后自增 x++ operator++(int)
		tmatrix_iterator operator++(int) { tmatrix_iterator yy = *this; ++*this; return yy; }
		// 虽然不知道有没有用但是写一下自减
		// 前自减 --x operator--()
		tmatrix_iterator& operator--() { x -= (y == 0); y = (y + (tm == nullptr ? tk->m : tm->m) - 1) % (tm == nullptr ? tk->m : tm->m); return *this; }
		// 后自减 x-- operator--(int)
		tmatrix_iterator operator--(int) { tmatrix_iterator yy = *this; --*this; return yy; }
		// 显然需要一个解引用
		_Valt& operator*() { return (*tm)[x][y]; }
		const _Valt& operator*() const { return (*tk)[x][y]; }
	};
public:
	// 默认构造函数，空矩阵
	type_matrix() : type_matrix(0, 0) {}
	// 构造函数，给出行列数
	type_matrix(size_t x, size_t y) { resize(x, y); }
	type_matrix(std::pair<size_t, size_t> xy) { resize(xy); }
	// 构造函数，从 valarray 或 vector 构造列向量
	type_matrix(const std::vector<_Valt>& x) { resize(x.size(), 1); for (unsigned i = 0; i < x.size(); i++) { _Val[i][0] = x[i]; } }
	type_matrix(const std::valarray<_Valt>& x) { resize(x.size(), 1); for (unsigned i = 0; i < x.size(); i++) { _Val[i][0] = x[i]; } }
	// 从 vector<vector<...>> 构造
	type_matrix(const std::vector<std::vector<_Valt>>& x) { resize(x.size(), x[0].size()); for (unsigned i = 0; i < x.size(); i++) { for (int j = 0; j < x[0].size(); j++) { _Val[i][j] = x[i][j]; } } }
	 //直接构造所有元素
	//type_matrix(const std::valarray<std::valarray<_Valt>> &x) { resize(x) }
	// 防止初始化列表报错
	type_matrix(const std::initializer_list<_Valt>& x) : type_matrix(static_cast<const std::valarray<_Valt>&>(x)) {}
	// 复制构造函数
	type_matrix(const type_matrix& r) : n(r.n), m(r.m), _Val(r._Val) { }
	// 移动构造函数
	// 翻阅 cppreference 可知，至少在目前的标准下，
	// 是有移动构造函数的。
	// https://zh.cppreference.com/w/cpp/numeric/valarray/valarray
	type_matrix(const type_matrix&& r) noexcept : n(r.n), m(r.m), _Val(r._Val) { }
	// 复制赋值运算符
	type_matrix& operator=(const type_matrix& y) { n = y.n; m = y.m; _Val = y._Val; return *this; }
	// 移动赋值运算符
	type_matrix& operator=(type_matrix&& y) noexcept { n = y.n; m = y.m; _Val = y._Val; return *this; }

	// 求大小
	std::pair<size_t, size_t> size() const { return { n, m }; }
	// 重置大小，顺便清空元素
	void resize(size_t x, size_t y) { n = x; m = y; _Val = std::valarray<std::valarray<_Valt>>(std::valarray<_Valt>(_Valt(), y), x); }
	void resize(std::pair<size_t, size_t> xy) { n = xy.first; m = xy.second; _Val = std::valarray<std::valarray<_Valt>>(std::valarray<_Valt>(0.0L, xy.second), xy.first); }
	// 求元素
	type_matrix<_Valt>::inner_vector operator[](size_t x) { return inner_vector(x, this); }
	const type_matrix<_Valt>::inner_vector operator[](size_t x) const { return inner_vector(x, this); }
	// 有时需要遍历所有元素
	tmatrix_iterator begin() { return tmatrix_iterator(this); }
	const tmatrix_iterator begin() const { return tmatrix_iterator(this); }
	tmatrix_iterator end() { return tmatrix_iterator(this, n, 0); }
	const tmatrix_iterator end() const { return tmatrix_iterator(this, n, 0); }
	
	// 类型转换
	operator _Valt() const
	{
		if (n != 1 || m != 1) throw std::length_error("Error in type_matrix::_Valt: The rows and cols of the matrix should be both 1.");
		return _Val[0][0];
	}
	operator std::vector<_Valt>() const
	{
		if (m != 1) throw std::length_error("Error in type_matrix::vector<_Valt>: The col of the matrix should be 1.");
		std::vector<_Valt> res;
		for (int i = 0; i < n; i++)
		{
			res.push_back(_Val[i][0]);
		}
		return res;
	}
	operator std::valarray<_Valt>() const
	{
		if (m != 1) throw std::length_error("Error in type_matrix::valarray<_Valt>: The col of the matrix should be 1.");
		std::valarray<_Valt> res;
		res.resize(n);
		for (unsigned i = 0; i < n; i++)
		{
			res[i] = _Val[i][0];
		}
		return res;
	}
};

// 比较运算符
namespace
{
	template<typename Q>
	bool operator==(const type_matrix<Q>& x, const type_matrix<Q>& y)
	{
		if (x.size() != y.size()) return false; // 应该抛出异常还是返回 false？数学意义上不等，但是实际上不应该大小不等？
		for (int i = 0; i < x.size().first; i++)
		{
			for (int j = 0; j < x.size().second; j++)
			{
				if (x[i][j] != y[i][j]) return false;
			}
		}
		return true;
	}
	template<typename Q>
	bool operator!=(const type_matrix<Q>& x, const type_matrix<Q>& y) { return !(x == y); }
}
// 数值运算符
namespace
{
	template<typename Q>
	type_matrix<Q> operator+(const type_matrix<Q>& x, const type_matrix<Q>& y)
	{
		if (x.size() != y.size()) throw std::invalid_argument("Error in operator+(const type_matrix &, const type_matrix &): The size(rows and columns) of the matrix x and y should be the same.");
		type_matrix<Q> res = x;
		for (size_t i = 0; i < x.size().first; i++)
		{
			for (size_t j = 0; j < y.size().second; j++)
			{
				res[i][j] += y[i][j];
			}
		}
		return res;
	}
	template<typename Q>
	type_matrix<Q> operator-(const type_matrix<Q>& x)
	{
		type_matrix res = x;
		for (size_t i = 0; i < res.size().first; i++)
		{
			for (size_t j = 0; j < res.size().second; j++)
			{
				res[i][j] = -res[i][j];
			}
		}
		return res;
	}
	template<typename Q>
	type_matrix<Q> operator-(const type_matrix<Q>& x, const type_matrix<Q>& y) { return x + (-y); }
	template<typename Q>
	type_matrix<Q> operator*(const type_matrix<Q>& x, const type_matrix<Q>& y)
	{
		if (x.size().second != y.size().first)
		{
			if (x.size() == y.size()) throw std::invalid_argument("Error in operator*(const type_matrix &, const type_matrix &): The columns of the matrix x and the rows of y should be the same. Maybe you mean dot product (use function dot_p to calculate)?");
			else throw std::invalid_argument("Error in operator*(const type_matrix &, const type_matrix &): The columns of the matrix x and the rows of y should be the same.");
		}
		size_t n = x.size().first, m = x.size().second, p = y.size().second;
		type_matrix<Q> res(n, p);
		for (unsigned i = 0; i < n; i++)
		{
			for (unsigned j = 0; j < p; j++)
			{
				for (unsigned k = 0; k < m; k++)
				{
					res[i][j] += x[i][k] * y[k][j]; // 哇和 Floyd 好像啊
				}
			}
		}
		return res;
	}
	template<typename Q>
	type_matrix<Q> operator*(const type_matrix<Q>& x, const Q& y)
	{
		auto res = x;
		for (auto& z : res) z *= y;
		return res;
	}
	template<typename Q>
	type_matrix<Q> operator*(const Q& y, const type_matrix<Q>& x) { return x * y; }
	template<typename Q>
	type_matrix<Q> operator/(const type_matrix<Q> &x, const Q &y)
	{
		auto res = x;
		for (auto& z : res) z /= y;
		return res;
	}
}
// 复合赋值运算符
namespace
{
	template<typename Q>
	type_matrix<Q>& operator+=(type_matrix<Q>& x, const type_matrix<Q>& y) { return x = x + y; }
	template<typename Q>
	type_matrix<Q>& operator-=(type_matrix<Q>& x, const type_matrix<Q>& y) { return x = x - y; }
	template<typename Q>
	type_matrix<Q>& operator*=(type_matrix<Q>& x, const type_matrix<Q>& y) { return x = x * y; }
	template<typename Q>
	type_matrix<Q>& operator*=(type_matrix<Q>& x, const Q& y) { return x = x * y; }
}
// 其它工具函数
namespace
{
	// 列向量（获取某一列）
	template <typename _Valt>
	type_matrix<_Valt> getcol(const type_matrix<_Valt>& p, size_t col) { size_t n = p.size().first; type_matrix<_Valt> res(n, 1); for (size_t i = 0; i < n; i++) { res[i][0] = p[i][col]; } return res; }
	// 以及行向量
	template <typename _Valt>
	type_matrix<_Valt> getrow(const type_matrix<_Valt>& p, size_t row) { size_t m = p.size().second; type_matrix<_Valt> res(1, m); for (size_t i = 0; i < m; i++) { res[0][i] = p[row][i]; } return res; }
	// 转置
	template<typename _Valt>
	type_matrix<_Valt> rotate(const type_matrix<_Valt>& p) { auto [n, m] = p.size(); type_matrix<_Valt> res(m, n); for (size_t i = 0; i < n; i++) { for (size_t j = 0; j < m; j++) { res[j][i] = p[i][j]; } } return res; }
	// 点积 Dot Product
	template<typename _Valt>
	type_matrix<_Valt> dot_p(const type_matrix<_Valt>& x, const type_matrix<_Valt>& y)
	{
		auto r = x.size();
		auto n = r.first, m = r.second;
		if (r != y.size()) throw std::invalid_argument("Error in Dot_P: The size of matrix x and y should be the same.");
		type_matrix<_Valt> res(n, m);
		for (size_t i = 0; i < n; i++)
		{
			for (size_t j = 0; j < m; j++)
			{
				res[i][j] = x[i][j] * y[i][j];
			}
		}
		return res;
	}
	// 进行同一运算（从左往右从上往下，可能需要满足结合律）
	template<typename _Valt, typename _Func>
	type_matrix<_Valt> all_op(const type_matrix<_Valt>& x, const _Func& f) { decltype(f(_Valt(), _Valt())) res{}; for (const auto& p : x) res = f(res, p); return res; }
	// 加法
	template<typename _Valt>
	type_matrix<_Valt> sum(const type_matrix<_Valt>& x) { return all_op(x, std::plus<_Valt>()); }
	// 乘法
	template<typename _Valt>
	type_matrix<_Valt> mul(const type_matrix<_Valt>& x) { return all_op(x, std::multiplies<_Valt>()); }
	// 外部扩大
	// 如，
	// A B
	// C D
	// 
	// 扩大 n=1, m=2 变为
	// 
	// A B A B
	// C D C D
	template<typename _Valt>
	type_matrix<_Valt> outer_expand(const type_matrix<_Valt>& x, size_t kx = 0, size_t ky = 0)
	{
		auto sz = x.size();
		type_matrix<_Valt> res(sz.first * kx, sz.second * ky);
		for (size_t i = 0; i < sz.first * kx; i++)
		{
			for (size_t j = 0; j < sz.second * ky; j++)
			{
				res[i] = x[i % sz.first][j % sz.second];
			}
		}
		return res;
	}
	// 内部扩大
	// 如，
	// A B
	// C D
	// 
	// 扩大 n=1, m=2 变为
	// 
	// A A B B
	// C C D D
	template<typename _Valt>
	type_matrix<_Valt> inner_expand(const type_matrix<_Valt>& x, size_t kx = 0, size_t ky = 0)
	{
		auto sz = x.size();
		type_matrix<_Valt> res(sz.first * kx, sz.second * ky);
		for (size_t i = 0; i < sz.first * kx; i++)
		{
			for (size_t j = 0; j < sz.second * ky; j++)
			{
				res[i] = x[i / kx][j / ky];
			}
		}
		return res;
	}
}

// 常用矩阵类
using matrix = type_matrix<long double>;