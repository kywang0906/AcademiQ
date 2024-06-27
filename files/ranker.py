"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer') -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords

    def query(self, query) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseduofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseduofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        query_parts = self.tokenize(query)
        if self.stopwords != None:
            for i, qword in enumerate(query_parts):
                if qword in self.stopwords:
                    query_parts[i] = None
        query_word_counts = Counter(query_parts)

        doc_word_counts_dict = defaultdict(dict)
        for qword in query_word_counts:
            if qword != None and qword in self.index.index:
                for docid, freq in self.index.index[qword]:
                    doc_word_counts_dict[docid][qword] = freq

        results = []
        for docid in doc_word_counts_dict:
            score = self.scorer.score(docid, doc_word_counts_dict[docid], query_word_counts)
            if score != 0:
                results.append((docid, score))
        if len(results) == 0:
            return []
        
        results.sort(key=lambda x: x[1], reverse=True)
    
        return results

class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError

class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        b = self.b
        k1 = self.k1
        k3 = self.k3
        d_count = self.index.get_statistics()['number_of_documents']
        d_length = self.index.get_doc_metadata(docid)['length']
        avdl = self.index.get_statistics()['mean_document_length']
        bm25_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            c_wq = query_word_counts[qword]
            idf = np.log((d_count-f_wd+0.5)/(f_wd+0.5))
            tf = (k1+1)*c_wd/(k1*(1-b+b*d_length/avdl)+c_wd)
            qtf = (k3+1)*c_wq/(k3+c_wq)
            bm25_score += idf * tf * qtf
        return bm25_score
    
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        b = self.b
        d_count  = self.index.get_statistics()['number_of_documents']
        d_length = self.index.get_doc_metadata(docid)['length']
        avdl = self.index.get_statistics()['mean_document_length']
        pn_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            c_wq = query_word_counts[qword]
            idf = np.log((d_count+1)/f_wd)
            tf = (1+np.log(1+np.log(c_wd)))/(1-b+b*d_length/avdl)
            qtf = c_wq
            pn_score += idf * tf * qtf
        return pn_score

class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        d_count  = self.index.get_statistics()['number_of_documents']
        tfidf_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            tf = np.log(c_wd+1)
            idf = np.log(d_count/f_wd) + 1
            tfidf_score += idf * tf
        return tfidf_score