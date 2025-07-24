import time
from datetime import datetime, timedelta

class GeminiRateLimiter:
    def __init__(self, rpm=5, tpm=250_000, rpd=100):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd

        self.requests = []
        self.tokens_this_minute = 0
        self.minute_start = datetime.now()
        self.daily_requests = []

    def wait_if_needed(self, tokens_needed):
        now = datetime.now()

        # Reset token counter every minute
        if now - self.minute_start >= timedelta(minutes=1):
            self.tokens_this_minute = 0
            self.requests = []
            self.minute_start = now

        # Clean old requests
        self.requests = [t for t in self.requests if now - t < timedelta(minutes=1)]
        self.daily_requests = [t for t in self.daily_requests if now - t < timedelta(days=1)]

        while len(self.requests) >= self.rpm or self.tokens_this_minute + tokens_needed > self.tpm or len(self.daily_requests) >= self.rpd:
            time.sleep(5)
            self.wait_if_needed(tokens_needed)  # Recheck recursively

    def register_request(self, tokens_used):
        now = datetime.now()
        self.tokens_this_minute += tokens_used
        self.requests.append(now)
        self.daily_requests.append(now)

def estimate_token_count(question, prompt_template):
    total_text = prompt_template.replace("{question}", question)
    return int(len(total_text.split()) * 1.5)