from llms_results_extraction.response_cleaner import ResponseCleaner

class AnswerChecker:
    """Checks if the model's response matches the gold answer."""

    def __init__(self, logger):
        self.logger = logger
        self.correct_answers = 0
        self.no_answer_count = 0
        self.label_answers_by_id = {}
        self.textual_answers_by_id = {}

    def check_answer(self, prompt: dict) -> None:
        """Validates the model's response against the gold answer."""
        gold_answer = ResponseCleaner.clean_text(prompt['gold_answer'])
        response = prompt['response']
    
        if prompt['ID'] not in self.label_answers_by_id:
            self.label_answers_by_id[prompt['ID']] = []
        
        if prompt['ID'] not in self.textual_answers_by_id:
            self.textual_answers_by_id[prompt['ID']] = []

        if response == 'no_answer':
            self.no_answer_count += 1
            self.label_answers_by_id[prompt['ID']].append(0)
            self.textual_answers_by_id[prompt['ID']].append('no_answer')
            self.logger.info(f'No answers for this prompt: {prompt["prompt"]}')
        elif response == gold_answer:
            self.correct_answers += 1
            self.label_answers_by_id[prompt['ID']].append(1)
            self.textual_answers_by_id[prompt['ID']].append(response)
        else:
            self.label_answers_by_id[prompt['ID']].append(0)
            self.textual_answers_by_id[prompt['ID']].append(response)
            self.logger.info(f'Incorrect prompt: {prompt["prompt"]}, '
                             f'response: {response}, gold_answer: {gold_answer}')

    def get_results(self):
        """Returns the current results as a dictionary."""
        return [self.correct_answers, self.label_answers_by_id, self.textual_answers_by_id]
        
