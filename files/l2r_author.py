from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, BM25, TF_IDF, PivotedNormalization

class L2RRanker:
    def __init__(self, document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.model = LambdaMART()
        self.feature_vectors_collection = []
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        X = []
        y = []
        qgroups = []

        for query in query_to_document_relevance_scores:
            query_parts = self.document_preprocessor.tokenize(query)
            if self.stopwords != None:
                for i, qword in enumerate(query_parts):
                    if qword in self.stopwords:
                        query_parts[i] = None
            doc_word_counts_dict = self.accumulate_doc_term_counts(self.feature_extractor.author_index, query_parts)

            qcount = 0
            for docid, RelevanceScore in query_to_document_relevance_scores[query]:
                X.append(self.feature_extractor.generate_features(
                    docid, 
                    doc_word_counts_dict[docid], 
                    query_parts,
                    query))
                y.append(RelevanceScore)
                qcount += 1
            qgroups.append(qcount)
        X = np.array(X)
        y = np.array(y)
        qgroups = np.array(qgroups)
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        word_counts_dict = defaultdict(dict)
        for qword in query_parts:
            if qword != None and qword in index.index:
                for docid, freq in index.index[qword]:
                    word_counts_dict[docid][qword] = freq
        return word_counts_dict

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """

        query_to_document_relevance_scores_train = defaultdict(list)
        train_df = pd.read_csv(training_data_filename)
        for i in tqdm(range(len(train_df))):
            query = train_df.iloc[i]['query']
            docid = train_df.iloc[i]['authorid']
            rel = train_df.iloc[i]['rel']
            query_to_document_relevance_scores_train[query].append((docid, rel))
        X_train, y_train, qgroups = self.prepare_training_data(query_to_document_relevance_scores_train)

        self.model.fit(X_train, y_train, qgroups, eval_at=[i for i in range(1, y_train.max() + 1)])

    def create_train_data(self, training_data_filename: str) -> None:
        query_to_document_relevance_scores_train = defaultdict(list)
        train_df = pd.read_csv(training_data_filename)
        for i in tqdm(range(len(train_df))):
            query = train_df.iloc[i]['query']
            docid = train_df.iloc[i]['authorid']
            rel = train_df.iloc[i]['rel']
            query_to_document_relevance_scores_train[query].append((docid, rel))
        X_train, y_train, qgroups = self.prepare_training_data(query_to_document_relevance_scores_train)
        return X_train, y_train, qgroups

    def fit(self, X_train, y_train, qgroups) -> None:
        self.model.fit(X_train, y_train, qgroups, eval_at=[i for i in range(1, y_train.max() + 1)])

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        return self.model.predict(X)
    
    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        cutoff = 1000
        query_parts = self.document_preprocessor.tokenize(query)
        if self.stopwords != None:
            for i, qword in enumerate(query_parts):
                if qword in self.stopwords:
                    query_parts[i] = None
        doc_word_counts_dict = self.accumulate_doc_term_counts(self.feature_extractor.author_index, query_parts)

        scores = self.ranker.query(query)
        scores_top = scores[0:cutoff]

        if len(scores_top) == 0:
            return []

        feature_vectors = []
        docid_top = []
        for score in scores_top:
            docid = score[0]
            docid_top.append(docid)
            ft_vec = self.feature_extractor.generate_features(
                    docid, doc_word_counts_dict[docid],
                    query_parts, query
                )
            feature_vectors.append(ft_vec)
            self.feature_vectors_collection.append(ft_vec)

        rerank_scores = self.predict(feature_vectors)
        self.feature_vectors = feature_vectors
        scores_resorted = []
        for i, docid in enumerate(docid_top):
            scores_resorted.append((docid, rerank_scores[i]))
        scores_resorted.sort(key=lambda x: x[1], reverse=True)
        
        scores_resorted += scores[cutoff:]
        return scores_resorted

class L2RFeatureExtractor:
    def __init__(self, author_index: InvertedIndex, author_on_title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 id_to_features: dict[int, dict[str, float]]) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            author_index: The inverted index for the contents of the author
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            id_to_features: A dictionary where the id is mapped to a dictionary
                with keys for feature names
        """
        self.author_index = author_index
        self.author_on_title_index = author_on_title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.id_to_features = id_to_features

    def get_text_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.author_index.get_doc_metadata(docid)['length']

    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        qtokens_count = Counter(query_parts)
        tf_score = 0
        for qword in qtokens_count:
            if qword == None or qword not in word_counts:
                continue
            c_wd = word_counts[qword]
            tf_score += np.log(c_wd+1)
        return tf_score

    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        scorer = TF_IDF(index)
        score = scorer.score(docid, word_counts, Counter(query_parts))
        return score

    def get_BM25_score(self, index, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        scorer = BM25(index)
        score = scorer.score(docid, doc_word_counts, Counter(query_parts))
        return score

    def get_pivoted_normalization_score(self, index, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        scorer = PivotedNormalization(index)
        score = scorer.score(docid, doc_word_counts, Counter(query_parts))
        return score

    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        try:
            pagerank = self.id_to_features[docid]['pagerank']
        except:
            pagerank = 0
        return pagerank
    
    def get_citation(self, docid: int) -> float:
        """
        Gets citation number for the given document.

        Args:
            docid: The id of the document

        Returns:
            citation number
        """
        try:
            citation = self.id_to_features[docid]['total_citations']
        except:
            citation = 0
        return citation
    
    def get_h_index(self, docid: int) -> float:
        """
        Gets h_index number for the given document.

        Args:
            docid: The id of the document

        Returns:
            h_index number
        """
        try:
            h_index = self.id_to_features[docid]['h_index']
        except:
            h_index = 0
        return h_index

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          query_parts: list[str], query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        feature_vector = []
    
        feature_vector.append(self.get_text_length(docid))

        feature_vector.append(len(query_parts))

        feature_vector.append(self.get_tf(self.author_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_tf_idf(self.author_index, docid, doc_word_counts, query_parts))
        
        feature_vector.append(self.get_tf(self.author_on_title_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_tf_idf(self.author_on_title_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_BM25_score(self.author_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_pivoted_normalization_score(self.author_index, docid, doc_word_counts, query_parts))
        
        feature_vector.append(self.get_BM25_score(self.author_on_title_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_pivoted_normalization_score(self.author_on_title_index, docid, doc_word_counts, query_parts))

        feature_vector.append(self.get_pagerank_score(docid))
        
        feature_vector.append(self.get_citation(docid))
        
        feature_vector.append(self.get_h_index(docid))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        self.default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            "n_jobs": -1,
            "verbosity": 1,
        }

        if params:
            self.default_params.update(params)

        self.ranker = lightgbm.LGBMRanker(
            objective = self.default_params['objective'],
            boosting_type = self.default_params['boosting_type'],
            n_estimators = self.default_params['n_estimators'],
            importance_type = self.default_params['importance_type'],
            metric= self.default_params['metric'],
            num_leaves = self.default_params['num_leaves'],
            learning_rate = self.default_params['learning_rate'],
            max_depth = self.default_params['max_depth'],
            n_jobs = self.default_params['n_jobs']
        )

    def re_init(self, params=None) -> None:
        
        if params:
            self.default_params.update(params)

        self.ranker = lightgbm.LGBMRanker(
            objective = self.default_params['objective'],
            boosting_type = self.default_params['boosting_type'],
            n_estimators = self.default_params['n_estimators'],
            importance_type = self.default_params['importance_type'],
            metric= self.default_params['metric'],
            num_leaves = self.default_params['num_leaves'],
            learning_rate = self.default_params['learning_rate'],
            max_depth = self.default_params['max_depth'],
            n_jobs = self.default_params['n_jobs']
        )

    def fit(self, X_train, y_train, qgroups_train, eval_at=[1,2,3,4,5]):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        self.ranker.fit(
            X=X_train,
            y=y_train,
            group=qgroups_train,
            eval_at=eval_at)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        pred_vals = self.ranker.predict(featurized_docs)
        return pred_vals