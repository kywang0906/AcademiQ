SCRACTCH_PATH = "/gpfs/accounts/stats_dept_root/stats_dept1/nawatsw/si699"
DATA_VERSION = "v16"
PAPER_DATA_PATH = f"{SCRACTCH_PATH}/DBLP-Citation-network-V15.1.json"
PAPER_ABSTRACT_INDEX = f"{SCRACTCH_PATH}/paper_DBLP_abstract_index"
PAPER_TITLE_INDEX = f"{SCRACTCH_PATH}/paper_DBLP_title_index"
AUTHOR_INDEX = f"{SCRACTCH_PATH}/author_index"
AUTHOR_ON_TITLE_INDEX = f"{SCRACTCH_PATH}/author_on_title_index"
RECOG_CATEGORY_PATH = f'{SCRACTCH_PATH}/recognized_categories.pickle'

DOC_CATEGORY_INFO_PATH = f'{SCRACTCH_PATH}/doc_category_info.pickle'
DOCID_LIST_PATH = f'{SCRACTCH_PATH}/docid_list.pickle'
DOCID_TO_ID_PATH = f'{SCRACTCH_PATH}/docid_to_id_dict.pickle'
DOCID_TO_YEAR_RELEASE_PATH = f'{SCRACTCH_PATH}/docid_to_year_release.pickle'
DOCID_TO_AUTHORID_PATH = f'{SCRACTCH_PATH}/docid_to_authorid.pickle'
DOCID_TO_NETWORK_FEATURES_PATH = f'{SCRACTCH_PATH}/docid_to_network_features.pickle'
DOCID_TO_CITATION_PATH = f'{SCRACTCH_PATH}/docid_to_citation.pickle'

AUTHORID_LIST_PATH = f'{SCRACTCH_PATH}/authorid_list.pickle'
AUTHORID_TO_ID_PATH = f'{SCRACTCH_PATH}/authorid_to_id_dict.pickle'
AUTHORID_TO_FEATURES_PATH = f'{SCRACTCH_PATH}/authorid_to_features.pickle'

AUTHOR_COLLECTION_PATH = f'{SCRACTCH_PATH}/author_collection.pickle'
AUTHORID_TO_AUTHOR_NAME_PATH = f'{SCRACTCH_PATH}/authorid_to_author_name.pickle'
PAPER_NETWORK_METRICS_PATH = f'{SCRACTCH_PATH}/various_metrics.pickle'

BM25_RANKER_PATH = f'{SCRACTCH_PATH}/BM25Ranker.pickle'
L2R_RANKER_PATH = f'{SCRACTCH_PATH}/l2rRanker.pickle'
L2R_RANKER_FITTED_PATH = f'{SCRACTCH_PATH}/l2rRanker_fitted.pickle'

AUTHOR_BM25_RANKER_PATH = f'{SCRACTCH_PATH}/author_BM25Ranker.pickle'
AUTHOR_L2R_RANKER_PATH = f'{SCRACTCH_PATH}/author_l2rRanker.pickle'
AUTHOR_L2R_RANKER_FITTED_PATH = f'{SCRACTCH_PATH}/author_l2rRanker_fitted.pickle'

TRAIN_PAPER_DATA_PATH = 'datasets/train_paper_data.csv'
TEST_PAPER_DATA_PATH = 'datasets/test_paper_data.csv'

TRAIN_AUTHOR_DATA_PATH = 'datasets/train_author_data.csvs'
TEST_AUTHOR_DATA_PATH = 'datasets/test_author_data.csv'
TRAIN_TEST_AUTHOR_DATA_PATH = 'datasets/train_test_author_data.csv'

STOPWORD_PATH = f"{SCRACTCH_PATH}/stopwords.txt"
BIENCODER_MODEL_NAME = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
TOTAL_PAPER_COUNT = 6404472
CATEGORIES_COUNT_CUTOFF = 2000