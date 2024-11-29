from llms_results_extraction.response_cleaner import ResponseCleaner

class ResponseProcessor:
    """Processes responses from different LLM models."""

    def __init__(self, model: str):
        self.model = model

    def process(self, prompt: dict) -> dict:
        """Processes the response based on the model type."""
        if self.model == 'gpt-4':
            return self._process_gpt_response(prompt)
        elif self.model == 'gemini':
            return self._process_gemini_response(prompt)
        else:
            return self._process_hg_response(prompt)

    def _process_gpt_response(self, prompt: dict) -> dict:
        """Extracts and processes GPT model responses."""
        prompt['response'] = ResponseCleaner.clean_text(prompt['response']['choices'][0]['message']['content'])
        return prompt

    def _process_gemini_response(self, prompt: dict) -> dict:
        """Extracts and processes Gemini model responses."""
        try:
            prompt['response'] = ResponseCleaner.clean_text(prompt['response']['candidates'][0]['content']['parts'][0]['text'])
        except KeyError:
            prompt['response'] = 'no_answer'
        return prompt

    def _process_hg_response(self, prompt: dict) -> dict:
        """Extracts and processes HG model responses."""
        prompt['response'] = prompt['cleaned_response']
        return prompt