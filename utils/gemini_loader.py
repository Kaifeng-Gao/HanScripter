import google.generativeai as genai
import yaml

class GeminiLoader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.api_key = self.config['access_token']['google_token']
        self.configure_api()
        self.model = self.create_model()
    
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def configure_api(self):
        genai.configure(api_key=self.api_key)
    
    def create_model(self):
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        return genai.GenerativeModel(
            model_name='gemini-pro',
            safety_settings=safety_settings
        )

# Usage
if __name__ == "__main__":
    config_path = 'config.yaml'
    model_loader = GeminiLoader(config_path)
    model = model_loader.model