import string

class ResponseCleaner:
    """Handles the cleaning of the prompt response."""

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Removes punctuation from a given text."""
        return ''.join([char for char in text if char not in string.punctuation])

    @staticmethod
    def clean_text(text: str) -> str:
        """Trims and converts a given text to lowercase."""
        return text.strip().lower()
