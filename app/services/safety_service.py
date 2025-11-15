import re
import logging

logger = logging.getLogger(__name__)

class SafetyNet:
    def __init__(self):
        # Improvement 2: Using word boundaries (this was already correct in your code)
        # This is for Improvement 1: Separating self-harm detection
        self.crisis_keywords = re.compile(
            r'\b(suicide|kill myself|kys|want to die)\b', 
            re.IGNORECASE
        )
        
        # This is for Improvement 1: Separating sexual/blocked content
        self.blocked_keywords = re.compile(
            r'\b(sex|porn|nude|erotic|fuck|bitch|shit)\b', 
            re.IGNORECASE
        )
        
        # Improvement 3: Allow-list for contextual checks
        # Keywords that, if present, might make a blocked word "safe"
        self.allow_keywords = re.compile(
            r'\b(addiction|therapy|discussing|help with|dealing with|problem about)\b',
            re.IGNORECASE
        )

    def check_crisis(self, text):
        """Checks for immediate crisis/self-harm language."""
        return bool(self.crisis_keywords.search(text))

    def check_blocked(self, text):
        """
        Checks for general explicit content, allowing for context.
        Returns True if BLOCKED, False if SAFE.
        (Note: Renamed from check_general_safety for clarity)
        """
        # Improvement 3: Contextual Check Logic
        
        # 1. Check if a blocked word is present
        if not self.blocked_keywords.search(text):
            return False  # Not blocked, safe to proceed
        
        # 2. A blocked word was found. NOW check for "allow" context.
        if self.allow_keywords.search(text):
            # e.g., "discussing porn addiction"
            return False  # Context makes it "safe", so it's NOT blocked
        
        # 3. Blocked word found and NO allow-context was found
        return True

    def get_refusal(self, text_for_logging):
        """
        Returns a refusal message for generally blocked content.
        
        Improvement 4: Logs the problematic message for monitoring.
        NOTE: Full anonymization (stripping PII) should happen
        in your main app before this class is even called, 
        if required by your privacy/legal standards.
        """
        logger.warning(f"Refusing to respond due to blocked content. Text: {text_for_logging}")
        return "Whoa there, let's keep things safe. Tell me something else?"

    def get_crisis_response(self, text_for_logging):
        """
        Returns a supportive crisis response.
        
        Improvement 1: Provides immediate supportive wording and helpline guidance.
        Improvement 4: Logs the crisis message for monitoring.
        """
        logger.warning(f"Crisis language detected, providing support response. Text: {text_for_logging}")
        
        # Provides generic, internationally-aware helpline information.
        # You can't reliably detect region, so providing common/global ones is best.
        return (
            "It sounds like you're going through an incredibly difficult time. "
            "Please know that you're not alone and there are people who want to help. "
            "You can connect with someone right now by calling or texting 988 in the US and Canada, "
            "or by calling 111 in the UK. For resources in other countries, "
            "you can visit findahelpline.com. Please reach out to them."
        )

# --- Example of how to use the new class ---
if __name__ == "__main__":
    
    # Setup a basic logger to see the output
    logging.basicConfig(level=logging.INFO)
    
    net = SafetyNet()
    
    # --- Test Cases ---
    
    # 1. Crisis Test
    crisis_text = "I really want to kill myself"
    print(f"Checking: '{crisis_text}'")
    if net.check_crisis(crisis_text):
        print(f"RESPONSE: {net.get_crisis_response(crisis_text)}\n")

    # 2. Blocked Test
    blocked_text = "show me some porn"
    print(f"Checking: '{blocked_text}'")
    # Note the new logic: elif check_blocked
    if not net.check_crisis(blocked_text) and net.check_blocked(blocked_text):
        print(f"RESPONSE: {net.get_refusal(blocked_text)}\n")

    # 3. Blocked Test with Allow-list (Improvement 3)
    allowed_text = "I need help with my porn addiction"
    print(f"Checking: '{allowed_text}'")
    is_crisis = net.check_crisis(allowed_text)
    is_blocked = net.check_blocked(allowed_text) # This should now be FALSE
    
    print(f"-> Is Crisis: {is_crisis}")
    print(f"-> Is Blocked: {is_blocked}")
    
    if not is_crisis and not is_blocked:
        print("RESPONSE: (This would go to the normal bot response)\n")
        
    # 4. Normal Text
    normal_text = "The weather is nice today"
    print(f"Checking: '{normal_text}'")
    is_crisis = net.check_crisis(normal_text)
    is_blocked = net.check_blocked(normal_text)
    print(f"-> Is Crisis: {is_crisis}")
    print(f"-> Is Blocked: {is_blocked}")
    if not is_crisis and not is_blocked:
        print("RESPONSE: (This would go to the normal bot response)\n")