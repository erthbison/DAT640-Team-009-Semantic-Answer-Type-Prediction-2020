import nltk
import re
import pickle
from typing import Iterable, List, Any, Tuple
from collections import Counter, UserDict
import json

from Classes import Entity

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def preprocess(doc: str) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]

def process_question_data(file):
    print("Processing file", file)
    with open(file, "r")  as f:
        data = json.load(f)
        question_list = []
        category_list = []
        type_list = []
        for question in data:
            if question["question"] is not None:
                question_list.append(question["question"])
            else:
                question_list.append("")
            category_list.append(question["category"])
            type_list.append(question["type"])
    return question_list, category_list, type_list

class InvertedIndex(UserDict):
    def __init__(self) -> None:
        self.data = {}

    # Divides the index into 2 dictionaries. One for each field.
    # The dictionary for each field then contains a nested dictionary. 
    # The first nested dictionary is keyed by the term
    # The second nested dictionary is keyed by the doc_id. The value is the frequency

    def get_postings(self, term: str) -> List[Any]:
        """Fetches the posting list for a given term.

        Args:
            term: Term for which to get postings.

        Returns:
            List of postings for the given term in the given field.
        """
        # TODO
        # A posting is a (doc_id, frequency) tuple. 
        # A list of these tuples is returned for the provided field and term
        entries = self.data.get(term, {})
        return [(doc_id, freq) for doc_id, freq in entries.items()]

    def get_num_postings(self, term:str) -> int:
        """Returns the number of postings for a given term
        
        Args:
            term: Term for which to get postings

        Returns:
            Number of postings for the given term
        """
        return len(self.data.get(term, {}))
        

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Return the frequency of a given term in a document.

        Args:
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        # TODO
        entries = self.data.get(term, {})
        return entries.get(doc_id, 0)

    def get_terms(self) -> List[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            Set of all terms in a given field.
        """
        # TODO
        return list(self.data.keys())

    def add_posting(self, term: str, doc_id: str, freq:int):
        # Retrieve the dictionary for this field and this term
        entries = self.data.get(term, {})
        # Update the dictionary for this term
        entries[doc_id] =  freq
        # Update the dictionaries storing the value
        self.data[term] = entries

class FieldedInvertedIndex(UserDict):
    def __init__(self) -> None:
        self.data = {}

    # Divides the index into 2 dictionaries. One for each field.
    # The dictionary for each field then contains a nested dictionary. 
    # The first nested dictionary is keyed by the term
    # The second nested dictionary is keyed by the doc_id. The value is the frequency

    def get_postings(self, field: str, term: str) -> List[Any]:
        """Fetches the posting list for a given field and term.

        Args:
            field: Field for which to get postings.
            term: Term for which to get postings.

        Returns:
            List of postings for the given term in the given field.
        """
        # TODO
        # A posting is a (doc_id, frequency) tuple. 
        # A list of these tuples is returned for the provided field and term
        entries = self.data.get(field, {}).get(term, {})
        return [(doc_id, freq) for doc_id, freq in entries.items()]

    def get_num_postings(self, field: str, term:str) -> int:
        """Returns the number of postings for a given term
        
        Args:
            field: Field for which to get postings.
            term: Term for which to get postings

        Returns:
            Number of postings for the given term
        """
        return len(self.data.get(field, {}).get(term, {}))
        

    def get_term_frequency(self, field: str, term: str, doc_id: str) -> int:
        """Return the frequency of a given term in a document.

        Args:
            field: Index field.
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        # TODO
        entries = self.data.get(field, {}).get(term, {})
        return entries.get(doc_id, 0)

    def get_terms(self, field: str) -> List[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            Set of all terms in a given field.
        """
        # TODO
        return list(self.data.get(field, {}).keys())

    def add_posting(self, term: str, doc_id: str, freq:int, field:str):
        # Retrieve the dictionary for this field and this term
        field_index = self.data.get(field, {})
        entries = field_index.get(term, {})
        # Update the dictionary for this term
        entries[doc_id] =  freq
        # Update the dictionaries storing the value
        field_index[term] = entries
        self.data[field] = field_index
    
def index_data(data: Iterable[Tuple[str, str]]) -> InvertedIndex:
    index = InvertedIndex()
    num_documents = len(data)
    for i, (id, element) in enumerate(data):
        if (i + 1) % (num_documents // 100) == 0:
            print(f"\r{round(100*(i/num_documents))}% indexed.", end="")
        terms = preprocess(element)
        ctr = Counter(terms)
        for term, freq in ctr.items():
            index.add_posting(term, id, freq)
    print("Indexing completed")
    return index

def index_entity(data: Iterable[Entity], fields: Iterable[str]) -> FieldedInvertedIndex():
    index = FieldedInvertedIndex()
    num_documents = len(data)
    for i, entity in enumerate(data):
        if (i + 1) % (num_documents // 100) == 0:
            print(f"\r{round(100*(i/num_documents))}% indexed.", end="")
        for field in fields:
            field_data = entity.__getattribute__(field)
            terms = preprocess(field_data)
            ctr = Counter(terms)
            for term, freq in ctr.items():
                index.add_posting(term, entity.name, freq, field)
    print("Indexing completed")
    return index


def preprocess_entity(e, i, total):
    if (i + 1) % (total // 100) == 0:
        print(f"\r{round(100*(i/total))}% processed...", end="")
    return preprocess(e.get_document())

if __name__ == "__main__":
    with open("types-entities.pkl", "rb") as f:
        _, entities = pickle.load(f)

    print("Started preprocessing data")


    # index = index_data([(e.name, e.get_document()) for e in entities])
    # print("Finished preprocessing data")
    # with open("Data/pickle/entity-index.pkl", "wb") as f:
    #     pickle.dump(index, f)
    # print("Finished pickling data")

    # index = index_entity(entities, ["name", "alternative_names"])
    # print("Finished preprocessing data")
    # with open("Data/pickle/entity-fielded-index.pkl", "wb") as f:
    #     pickle.dump(index, f)
    # print("Finished pickling data")

    index = list(map(preprocess_entity, entities, range(len(entities)), [len(entities)] * len(entities)))
    print("\nFinished preprocessing data")
    with open("Data/pickle/entity-term-list.pkl", "wb") as f:
        pickle.dump(index, f)
    print("Finished pickling data")
    

    # test_data, test_category, test_type = process_question_data("Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_test.json")
    # train_data, train_category, train_type = process_question_data("Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_train.json")

    # with open("Data/pickle/train_data.pkl", "wb") as file:
    #     pickle.dump((train_quesry, train_category, train_type), file)
    # with open("Data/pickle/test_data.pkl", "wb") as file:
    #     pickle.dump((test_query, test_category, test_type), file)
    # with open("Data/pickle/entities.pkl", "wb") as file:
    #     pickle.dump(entities, file)