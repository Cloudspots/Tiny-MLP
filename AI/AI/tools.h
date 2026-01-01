#pragma once
#include <concepts>
#include <algorithm>

// （C++20）concept 约束，用于实现中间层
template<typename T>
concept has_beginend = requires(T k) { { std::begin(k) } -> std::same_as<decltype(std::end(k))>; };

// 遍历所有元素
// 普通模板：没有 begin() 和 end()，直接调用
template<typename T, typename _Fun>
auto forall(const T& x, const _Fun& y) { return y(x); }
// 特化模版：有 begin() 和 end()，范围 for 循环调用
// 顺便，这里甚至可以嵌套！如 forall(type_matrix<type_matrix<long double>>, xxx)
template<has_beginend T, typename _Fun>
auto forall(const T& x, const _Fun& y) { T res = x; for (auto& p : res) p = forall(p, y); return res; }
