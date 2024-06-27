import math
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

def map_score(search_result_relevances: list[int], cut_off=20) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    r_count = 0
    ap = []
    for i, r in enumerate(search_result_relevances):
        if r == 1:
            if i < cut_off:
                r_count += 1
                ap.append(r_count/(i+1))
            else:
                ap.append(0)

    if len(ap) == 0:
        return 0
    
    return np.mean(ap)


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=20):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    if len(search_result_relevances) == 0:
        return 0
    dcg = search_result_relevances[0] + sum([x/np.log10(i+2) for i, x in enumerate(search_result_relevances[1:cut_off])])
    idcg = ideal_relevance_score_ordering[0] + sum([x/np.log10(i+2) for i, x in enumerate(ideal_relevance_score_ordering[1:cut_off])])

    return dcg/idcg if idcg !=0 else 0

def save_query_result(relevance_data_filename, ranker, save_path):
    rel_df = pd.read_csv(relevance_data_filename)
    query_list = rel_df['query'].unique()
    rank_result_dict = {}
    for query in tqdm(query_list):
        rank_result_dict[query] = ranker.query(query)
    with open(save_path, 'wb') as f:
        pickle.dump(rank_result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def run_relevance_tests(relevance_data_filename, rank_result_path, id_col='docid') -> dict[str, float]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    
    with open(rank_result_path, 'rb') as f:
        rank_result_dict = pickle.load(f)
    rel_df = pd.read_csv(relevance_data_filename)
    query_list = rel_df['query'].unique()
    map_list = []
    ndcg_list = []
    for query in tqdm(query_list):
        if query not in rank_result_dict:
            continue
        rank_result = rank_result_dict[query]
        actual_rel = []
        rel_selected_df = rel_df[rel_df['query'] == query]
        rel_docid_list = list(rel_selected_df[id_col])

        for result in rank_result:
            if result[0] in rel_docid_list:
                actual_rel.append(rel_selected_df[rel_selected_df[id_col] == result[0]].iloc[0]["rel"])
            else:
                actual_rel.append(1)
        
        map_input_rel = [0 if x < 2 else 1 for x in actual_rel]
        map_list.append(map_score(map_input_rel))

        ideal_rel = sorted(actual_rel, reverse=True)
        ndcg_list.append(ndcg_score(actual_rel, ideal_rel))

    return {'map': map_list, 'ndcg': ndcg_list}