from llms_results_extraction.response_processor import ResponseProcessor
from llms_results_extraction.answer_checker import AnswerChecker
from llms_results_extraction.response_cleaner import ResponseCleaner

class LLMSResultsExtraction:
    """Main class for extracting results from the model responses."""

    def __init__(self, wrong_answers_logger):
        self.wrong_answers_logger = wrong_answers_logger
        self.all_responses = []

    def extract_results(self, data: list, model: str = 'gpt-4'):
        """Extracts and processes results from the data."""
        processor = ResponseProcessor(model)
        answer_checker = AnswerChecker(self.wrong_answers_logger)
    
        for prompt in data:
            # Process and clean responses
            prompt = processor.process(prompt)
            prompt['response'] = ResponseCleaner.remove_punctuation(prompt['response']).lower()
            self.all_responses.append(prompt['response'])
        
            # Check the response accuracy
            answer_checker.check_answer(prompt)
            correct_answers, label_answers_by_id, textual_answers_by_id = answer_checker.get_results()

        # Return combined results
        return  correct_answers, label_answers_by_id, self.all_responses, textual_answers_by_id

