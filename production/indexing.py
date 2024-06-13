from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from document_preprocessor import Tokenizer
import re

class IndexType(Enum):
    InvertedIndex = 'BasicInvertedIndex'

class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = defaultdict(Counter)  # Central statistics of the index
        self.index = {}  # Index
        self.document_metadata = {}  # Metadata like length, number of unique tokens of the documents

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # TODO: Save the index files to disk
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['stored_total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0
        self.index = defaultdict(list)
        
    def index_search(self, posting, docid):
        low = 0
        high = len(posting) - 1
        insert_idx = -1
        if posting[high][0] == docid:
            return high, True
        while low <= high:
            mid = (low + high) // 2
            mid_value = posting[mid][0]
            if mid_value == docid:
                return mid, True
            elif mid_value < docid:
                low = mid + 1
                insert_idx = low
            else:
                high = mid - 1 
                insert_idx = mid
        return insert_idx, False

    def remove_doc(self, docid: int) -> None:
        removed_token_rec = []
        for token in self.index:
            idxRmv, foundFlag = self.index_search(self.index[token], docid)
            if foundFlag:
                self.statistics['stored_total_token_count'] -= self.index[token][idxRmv][1]
                self.index[token].pop(idxRmv)
                if len(self.index[token]) == 0:
                    removed_token_rec.append(token)

        for token in removed_token_rec:
            del self.index[token]
        self.statistics['total_token_count'] -= self.document_metadata[docid]['length']
        self.statistics['number_of_documents'] -= 1
        self.statistics['unique_token_count'] = len(self.index)
        self.statistics['mean_document_length'] = self.statistics['total_token_count']/self.statistics['number_of_documents'] if self.statistics['number_of_documents'] != 0 else 0
        del self.document_metadata[docid]


    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        self.document_metadata[docid] = defaultdict(int)
        self.document_metadata[docid]['length'] = len(tokens)
        self.statistics['total_token_count'] += len(tokens)
        self.statistics['number_of_documents'] += 1
        for token in tokens:
            if token == None:
                continue
            if token not in self.index: # no token in index
                self.index[token] = [(docid, 1)]
                self.statistics['stored_total_token_count'] += 1
                self.document_metadata[docid]['unique_tokens'] += 1
                continue
            if docid > self.index[token][-1][0]: # new token in doc
                self.index[token].append((docid, 1))
                self.statistics['stored_total_token_count'] += 1
                self.document_metadata[docid]['unique_tokens'] += 1
                continue
            idxAdd, foundFlag = self.index_search(self.index[token], docid)
            if foundFlag: # existing token in doc
                self.index[token][idxAdd] = (docid, self.index[token][idxAdd][1]+1)
                self.statistics['stored_total_token_count'] += 1
            else: # new token in doc
                self.index[token].insert(idxAdd, (docid, 1))
                self.statistics['stored_total_token_count'] += 1
                self.document_metadata[docid]['unique_tokens'] += 1
        self.statistics['unique_token_count'] = len(self.index)
        self.statistics['mean_document_length'] = self.statistics['total_token_count']/self.statistics['number_of_documents'] if self.statistics['number_of_documents'] != 0 else 0


    def add_inidx_doc(self, docid: int, invertedIndex: dict[str, list], indexLength: int) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        self.document_metadata[docid] = defaultdict(int)
        self.document_metadata[docid]['length'] = indexLength
        self.statistics['total_token_count'] += indexLength
        self.statistics['number_of_documents'] += 1
        for token in invertedIndex:
            if token not in self.index: # no token in index
                self.index[token] = [(docid, len(invertedIndex[token]))]
                self.statistics['stored_total_token_count'] += len(invertedIndex[token])
                self.document_metadata[docid]['unique_tokens'] += len(invertedIndex[token])
                continue
            if docid > self.index[token][-1][0]: # new token in doc
                self.index[token].append((docid, len(invertedIndex[token])))
                self.statistics['stored_total_token_count'] += len(invertedIndex[token])
                self.document_metadata[docid]['unique_tokens'] += len(invertedIndex[token])
                continue
            idxAdd, foundFlag = self.index_search(self.index[token], docid)
            if foundFlag: # existing token in doc
                self.index[token][idxAdd] = (docid, self.index[token][idxAdd][1]+len(invertedIndex[token]))
                self.statistics['stored_total_token_count'] += len(invertedIndex[token])
            else: # new token in doc
                self.index[token].insert(idxAdd, (docid, len(invertedIndex[token])))
                self.statistics['stored_total_token_count'] += len(invertedIndex[token])
                self.document_metadata[docid]['unique_tokens'] += len(invertedIndex[token])
        self.statistics['unique_token_count'] = len(self.index)
        self.statistics['mean_document_length'] = self.statistics['total_token_count']/self.statistics['number_of_documents'] if self.statistics['number_of_documents'] != 0 else 0


    def get_postings(self, term: str) -> list[tuple[int, int]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        term_metadata = {}
        term_metadata['count'] = sum([post[1] for post in self.index[term]])
        term_metadata['document_count'] = len(self.index[term])
        return term_metadata

    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        return self.statistics

    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        if not os.path.exists(f'{index_directory_name}'):
            os.mkdir(f'{index_directory_name}')
        with open(f'{index_directory_name}/index', "w") as outfile:
            json.dump(self.index, outfile)
        with open(f'{index_directory_name}/document_metadata', "w") as outfile:
            json.dump(self.document_metadata, outfile)
        with open(f'{index_directory_name}/statistics', "w") as outfile:
            json.dump(self.statistics, outfile)

    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        with open(f'{index_directory_name}/index', "r") as json_file:
            self.index = defaultdict(list)
            preindex = json.load(json_file)
            for term in tqdm(preindex):
                self.index[term] = [(int(docid), int(freq)) for docid, freq in preindex[term]]

        with open(f'{index_directory_name}/document_metadata', "r") as json_file:
            predocument_metadata = json.load(json_file)
            self.document_metadata = {}
            for docid in predocument_metadata:
                docid_int = int(docid)
                self.document_metadata[docid_int] = defaultdict(int)
                for value_name in predocument_metadata[docid]:
                    self.document_metadata[docid_int][value_name] = predocument_metadata[docid][value_name]

        with open(f'{index_directory_name}/statistics', "r") as json_file:
            self.statistics = json.load(json_file)

class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str], docid_to_id,
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict = None) -> InvertedIndex:
        """
        Creates an inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text
                and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index
        """
        if index_type == IndexType.InvertedIndex:
            index = BasicInvertedIndex()                

        with open(dataset_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                doc = json.loads(line)
                if doc['abstract'] == '' or doc['n_citation'] <=20:
                    continue
                
                text = doc[text_key]
                tokens = document_preprocessor.tokenize(text)
                
                docid = docid_to_id[doc['id']]

                if doc_augment_dict != None:
                    aug_text = " ".join(doc_augment_dict[docid])
                    aug_tokens = document_preprocessor.tokenize(aug_text)
                    tokens += aug_tokens

                if stopwords != None:
                    for i, token in enumerate(tokens):
                        if token in stopwords:
                            tokens[i] = None

                index.add_doc(docid, tokens)

            if minimum_word_frequency > 1:
                term_removed = []
                for token in index.index:
                    if index.get_term_metadata(token)['count'] < minimum_word_frequency:
                        term_removed.append(token)

                for token in term_removed:
                    for post in index.index[token]:
                        index.document_metadata[post[0]]['unique_tokens'] -= 1
                    index.statistics['stored_total_token_count'] -= index.get_term_metadata(token)['count']
                    index.statistics['unique_token_count'] -= 1
                    del index.index[token]
            
        return index

    @staticmethod
    def create_index_from_inverted_index(index_type: IndexType, dataset_path: str, stopwords: set[str], docid_to_id,
                     minimum_word_frequency: int, max_docs: int = -1) -> InvertedIndex:
        """
        Creates an inverted index from inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        """
        if index_type == IndexType.InvertedIndex:
            index = BasicInvertedIndex()                

        with open(dataset_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                doc = json.loads(line)
                if doc['abstract'] == '':
                    continue

                docid = doc['id']
                invertedIndex = doc['indexed_abstract']['InvertedIndex']
                indexLength = doc['indexed_abstract']['IndexLength']

                # reconstruct raw text from inverted index
                raw_tokens = [''] * indexLength
                for token in invertedIndex:
                    for pos in invertedIndex[token]:
                        raw_tokens[pos] = token

                # sanitize token
                edited_invertedIndex = {}
                for token in invertedIndex:
                    edited_token = re.sub(r"[,.;@#?!&$\(\)\{\}\[\]]+", "", token.lower())
                    edited_invertedIndex[edited_token] = invertedIndex[token]

                if stopwords != None:
                    token_to_rmv = []
                    for token in edited_invertedIndex:
                        if token in stopwords:
                            token_to_rmv.append(token)
                    for token in token_to_rmv:
                        del edited_invertedIndex[token]
                            
                index.add_inidx_doc(docid, edited_invertedIndex, indexLength)

            if minimum_word_frequency > 1:
                term_removed = []
                for token in index.index:
                    if index.get_term_metadata(token)['count'] < minimum_word_frequency:
                        term_removed.append(token)

                for token in term_removed:
                    for post in index.index[token]:
                        index.document_metadata[post[0]]['unique_tokens'] -= 1
                    index.statistics['stored_total_token_count'] -= index.get_term_metadata(token)['count']
                    index.statistics['unique_token_count'] -= 1
                    del index.index[token]
            
        return index


