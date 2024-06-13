from nltk.tokenize import RegexpTokenizer

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        output_tokens = []
        if self.lowercase == True:
            for token in input_tokens:
                output_tokens.append(token.lower())
            return output_tokens
        else:
            return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')
    
class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        preTokenList = self.tokenizer.tokenize(text)
        self.tokenList = self.postprocess(preTokenList)
        return self.tokenList