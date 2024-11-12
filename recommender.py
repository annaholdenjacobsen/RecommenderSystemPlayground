import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import matplotlib.pyplot as plt
import kagglehub

# Load the data
ml100k = ML100K('data/ml-100k')
ratings = ml100k.ratings
ratings.head()

algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)

def evaluate(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

if __name__ == '__main__':
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(evaluate('ItemItem', algo_ii, train, test))
        all_recs.append(evaluate('ALS', algo_als, train, test))

    all_recs = pd.concat(all_recs, ignore_index=True)
    print(all_recs.head())

    test_data = pd.concat(test_data, ignore_index=True)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    print(results.head())

    print(results.groupby('Algorithm').ndcg.mean())

    ndgc = results.groupby('Algorithm').ndcg.mean()

    ndgc.plot.bar()
    

    plt.show()

