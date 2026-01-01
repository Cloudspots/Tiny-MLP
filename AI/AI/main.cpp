#include "MLP.h"

int main()
{
	using namespace std;
	auto mlp = MLP<long double>({ 2, 1 });
	mt19937_64 mt(random_device{}());
	constexpr unsigned batches = 100;
	constexpr unsigned dataperbatch = 10;
	constexpr unsigned countperbatch = 1000;
	long double beta = 0.01;
	//mlp.set_function([](const matrix& x, const matrix& y) { return loss_func::MAE<long double>(x, y); }, [](const matrix& x, const matrix& y) { return loss_func::d_MAE<long double>(x, y); }, [](const matrix& x) { return x; }, [](const matrix& x) { matrix res; res.resize(x.size()); for (auto& x : res) x = 1; return res; });
	mlp.set_acf([](const matrix& in) { return in; }, [](const matrix& out) { matrix res; res.resize(out.size()); for (auto& x : res) x = 1; return res; });
	vector<vector<pair<long double, long double>>> bch;
	uniform_real_distribution<long double> ui(-1, 1);
	for (unsigned i = 1; i <= batches; i++)
	{
		bch.push_back({});
		for (unsigned j = 1; j <= dataperbatch; j++)
		{
			bch.back().push_back({ ui(mt), ui(mt) });
		}
	}
	for (unsigned i = 0; i < batches; i++)
	{
		long double totalloss = 0;
		for (unsigned cc = 0; cc < countperbatch; cc++)
		{
			vector<type_matrix<long double>> sdw, sdb;
			for (unsigned j = 0; j < dataperbatch; j++)
			{
				//int x = ui(mt), y = ui(mt);
				long double x = bch[i][j].first, y = bch[i][j].second;
				auto [loss, dw, db, da] = mlp.train({ x, y }, { x + y });
				if (j == 1)
				{
					sdw = dw;
					sdb = db;
				}
				else
				{
					for (unsigned k = 0; k < sdw.size(); k++)
					{
						sdw[k] += dw[k];
						sdb[k] += db[k];
					}
				}
				totalloss += loss;
			}
			for (auto& x : sdw) x = x / (long double)dataperbatch;
			for (auto& x : sdb) x = x / (long double)dataperbatch;
			mlp.apply_train(beta, sdw, sdb);
		}
		totalloss /= countperbatch;
		printf("Epoch: %d/%d. Average Loss: %.10Lf\n", i, batches, totalloss);
	}
	printf("Training finished!\n");
	while (true)
	{
		int x, y;
		scanf("%d%d", &x, &y);
		printf("%d + %d = %.10Lf\n", x, y, (long double)(mlp.get({ (long double)x / 1000000000, (long double)y / 1000000000 }).back()[0][0]) * 1000000000);
	}
	return 0;
}
// Link-Cut Tree
// LCT
// Binary Search Tree
// BST
// Size Balanced Tree
// **T
// Tree + Heap
// Treap
// Segment Tree
// ST (x
// Sparse Tablet(x Table
// ST (v