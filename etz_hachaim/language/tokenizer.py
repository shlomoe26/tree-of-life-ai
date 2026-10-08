class CharTokenizer:
    """
    Very simple character-level tokenizer.
    It reads the text, finds all unique characters, and builds a dictionary
    to convert text to numbers and back.
    """
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # String to integer (encoding)
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer to string (decoding)
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        """ Convert a string to a list of integers """
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """ Convert a list of integers to a string """
        return ''.join([self.itos[i] for i in l])
