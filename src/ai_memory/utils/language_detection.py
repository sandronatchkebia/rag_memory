"""Language detection and text normalization utilities."""

import re
from typing import Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

# Constants from the working ai_me repository
GEORGIAN_UNICODE_RANGE = re.compile(r'[\u10A0-\u10FF]')

ENGLISH_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his', 'how',
    'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'way', 'year', 'good', 'look', 'take',
    'time', 'work', 'back', 'come', 'give', 'know', 'life', 'make', 'most', 'over', 'some', 'than', 'them', 'very', 'when', 'will', 'with', 'your', 'about', 'after',
    'again', 'could', 'first', 'great', 'other', 'should', 'think', 'through', 'water', 'where', 'which', 'world', 'would', 'always', 'because', 'between', 'different',
    'everything', 'something', 'sometimes', 'together', 'without', 'questions', 'should', 'reschedule', 'wants', 'poor', 'remain', 'rich', 'richer', 'michael', 'socialist',
    'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'thanks', 'thank', 'please', 'sorry', 'excuse', 'me', 'yes', 'no', 'ok',
    'okay', 'sure', 'fine', 'great', 'bad', 'nice', 'cool', 'awesome', 'amazing', 'wow', 'oh', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'that', 'this',
    'these', 'those', 'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'office', 'meeting', 'call', 'email', 'message', 'text', 'phone', 'computer',
    'internet', 'website', 'app', 'program', 'food', 'drink', 'water', 'coffee', 'tea', 'breakfast', 'lunch', 'dinner', 'snack', 'restaurant', 'cafe', 'bar', 'hotel',
    'travel', 'trip', 'vacation', 'holiday', 'people', 'person', 'child', 'baby', 'parent', 'mother', 'father', 'sister', 'brother', 'son', 'daughter'
}

GEORGIAN_STOPWORDS = {
    'ra', 'aris', 'da', 'me', 'shen', 'es', 'is', 'gogo', 'bavshvi', 'mama', 'deda', 'dzma', 'gamarjoba', 'nakhvamdis', 'madloba', 'gagimarjos',
    'didi', 'patara', 'lamazi', 'tsotskhali', 'kargi', 'sakartvelo', 'tbilisi', 'batumi', 'kutaisi', 'rustavi', 'rogor', 'khar', 'kargad', 'tsudad'
}

GEORGIAN_CLUSTERS = [
    'sh', 'ch', 'ts', 'dz', 'kh', 'gh', 'ph', 'th', 'zh', 'q'
]


def detect_language(text: str) -> str:
    """Detect the language of text content using improved logic."""
    if not text or len(text.strip()) < 2:
        return "unknown"
    
    # Use the improved classification logic
    classification = classify_text(text)
    
    # Map classifications to language codes
    if classification == "ka":
        return "ka"  # Georgian Unicode
    elif classification == "ka_en":
        return "ka"  # Georgian Latin script
    elif classification == "en":
        return "en"  # English
    else:
        return "unknown"


def classify_text(text: str) -> str:
    """
    Classify text into three buckets:
    1. Georgian (Unicode): Contains Georgian characters
    2. Georgian (English Keyboard): Romanized Georgian
    3. English: Clean English
    
    Improved logic to better distinguish English vs Georgian Latin script
    """
    if not text or len(text.strip()) < 2:
        return "unknown"
    
    text = text.strip()
    
    # 0) Easy win: Georgian Unicode characters
    if GEORGIAN_UNICODE_RANGE.search(text):
        return "ka"
    
    # 1) Check if ASCII-only
    if not text.isascii():
        return "unknown"  # Contains non-ASCII but not Georgian
    
    # 2) ASCII-only: Decide English vs Romanized Georgian
    text_lower = text.lower()
    words = text_lower.split()
    
    # Handle very short texts first
    if len(words) == 1:
        word = words[0]
        if word in GEORGIAN_STOPWORDS:
            return "ka_en"
        elif word in {'ok', 'yes', 'no', 'hi', 'hey', 'bye', 'cool', 'nice', 'good', 'bad', 'wow', 'oh', 'ah', 'um', 'uh'}:
            return "en"
        else:
            # default to English for single unknown words to avoid dropping
            return "en"
    elif len(words) < 2:
        # default to English rather than unknown to maximize retention
        return "en"
    
    # STRICT ENGLISH DETECTION - Must pass multiple criteria
    
    # Criterion 1: High English dictionary hit-rate
    english_word_count = sum(1 for word in words if word in ENGLISH_WORDS)
    english_ratio = english_word_count / len(words)
    
    # Criterion 2: English pattern matches
    english_patterns = [
        r'\b(hi|hello|hey|good|morning|afternoon|evening|night|day|week|month|year)\b',
        r'\b(thanks|thank|you|please|sorry|excuse|me|yes|no|ok|okay|sure|fine|great|good|bad|nice|cool|awesome|amazing|wow|oh|hey|what|how|when|where|why|who|which|that|this|these|those|here|there|now|then|today|tomorrow|yesterday)\b',
        r'\b(work|home|school|university|college|office|meeting|call|email|message|text|phone|computer|internet|website|app|program|food|drink|water|coffee|tea|breakfast|lunch|dinner|snack|restaurant|cafe|bar|hotel|travel|trip|vacation|holiday|family|friend|people|person|man|woman|boy|girl|child|baby|parent|mother|father|sister|brother|son|daughter)\b'
    ]
    
    pattern_matches = 0
    for pattern in english_patterns:
        matches = re.findall(pattern, text_lower)
        pattern_matches += len(matches)
    
    # Criterion 3: Check for English sentence structure
    english_structure_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
    
    structure_word_count = sum(1 for word in words if word in english_structure_words)
    structure_ratio = structure_word_count / len(words)
    
    # ENGLISH CLASSIFICATION: More lenient - pass 2 out of 3 criteria
    english_score = 0
    if english_ratio >= 0.4:  # Lowered from 0.6
        english_score += 1
    if pattern_matches >= len(words) * 0.2:  # Lowered from 0.3
        english_score += 1
    if structure_ratio >= 0.15:  # Lowered from 0.2
        english_score += 1
    
    # Classify as English if it passes 2 out of 3 criteria
    if english_score >= 2:
        return "en"
    
    # GEORGIAN LATIN SCRIPT DETECTION - More specific signals
    
    # Signal 1: Georgian character clusters
    georgian_cluster_count = 0
    for word in words:
        # Skip very common English words that might contain these clusters
        if word in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'way', 'year', 'good', 'look', 'take', 'time', 'work', 'back', 'come', 'give', 'know', 'life', 'make', 'most', 'over', 'some', 'than', 'them', 'very', 'when', 'will', 'with', 'your', 'about', 'after', 'again', 'could', 'first', 'great', 'other', 'should', 'think', 'through', 'water', 'where', 'which', 'world', 'would', 'always', 'because', 'between', 'different', 'everything', 'something', 'sometimes', 'together', 'without', 'questions', 'should', 'reschedule', 'wants', 'poor', 'remain', 'rich', 'richer', 'michael', 'socialist'}:
            continue
            
        for cluster in GEORGIAN_CLUSTERS:
            if cluster in word:
                georgian_cluster_count += 1
                break
    
    cluster_ratio = georgian_cluster_count / len(words)
    if cluster_ratio >= 0.5:  # Lowered from 0.6
        return "ka_en"
    
    # Signal 2: Georgian word endings
    ending_count = 0
    for word in words:
        # Skip very common English words that might have these endings
        if word in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'way', 'year', 'good', 'look', 'take', 'time', 'work', 'back', 'come', 'give', 'know', 'life', 'make', 'most', 'over', 'some', 'than', 'them', 'very', 'when', 'will', 'with', 'your', 'about', 'after', 'again', 'could', 'first', 'great', 'other', 'should', 'think', 'through', 'water', 'where', 'which', 'world', 'would', 'always', 'because', 'between', 'different', 'everything', 'something', 'sometimes', 'together', 'without', 'questions', 'should', 'reschedule', 'wants', 'poor', 'remain', 'rich', 'richer', 'michael', 'socialist'}:
            continue
            
        # Check for Georgian word endings
        if any(word.endswith(ending) for ending in ['i', 'a', 'o', 'u', 'e']):
            ending_count += 1
    
    ending_ratio = ending_count / len(words)
    if ending_ratio >= 0.6:  # Lowered from 0.7
        return "ka_en"
    
    # Signal 3: Georgian stopwords
    stopword_count = sum(1 for word in words if word in GEORGIAN_STOPWORDS)
    stopword_ratio = stopword_count / len(words)
    if stopword_ratio >= 0.25:  # Lowered from 0.3
        return "ka_en"
    
    # Default to English for longer texts if no clear Georgian signals
    if len(words) >= 3:  # Lowered from 5
        # For longer texts, check if there are any Georgian signals before defaulting to English
        if any(word in GEORGIAN_STOPWORDS for word in words):
            return "ka_en"
        else:
            return "en"
    
    # Default to English for shorter texts
    return "en"


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalize text for processing."""
    if not text:
        return ""
    
    # Basic normalization
    text = text.strip()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Handle Georgian text (both keyboard layouts)
    if language == "ka" or is_georgian(text):
        # Already in Georgian script
        pass
    elif is_romanized_georgian(text):
        # Convert romanized Georgian to proper Georgian
        text = normalize_georgian_roman(text)
    
    return text


def is_georgian(text: str) -> bool:
    """Check if text contains Georgian characters."""
    if not text:
        return False
    
    # Georgian Unicode range: U+10A0-U+10FF
    georgian_chars = set("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
    return any(char in georgian_chars for char in text)


def is_romanized_georgian(text: str) -> bool:
    """Check if text is Georgian written with Roman characters."""
    if not text or len(text.strip()) < 3:
        return False
    
    # Use the improved classification
    classification = classify_text(text)
    return classification == "ka_en"


def normalize_georgian_roman(text: str) -> str:
    """Normalize Georgian text written with Roman characters."""
    if not text:
        return text
    
    # Common romanized Georgian to Georgian mappings
    roman_to_georgian = {
        # Vowels
        'a': 'ა', 'e': 'ე', 'i': 'ი', 'o': 'ო', 'u': 'უ',
        # Consonants
        'b': 'ბ', 'g': 'გ', 'd': 'დ', 'v': 'ვ', 'z': 'ზ',
        't': 'თ', 'k': 'კ', 'l': 'ლ', 'm': 'მ', 'n': 'ნ',
        'p': 'პ', 'zh': 'ჟ', 'r': 'რ', 's': 'ს', 't': 'ტ',
        'p': 'ფ', 'k': 'ქ', 'gh': 'ღ', 'q': 'ყ', 'sh': 'შ',
        'ch': 'ჩ', 'ts': 'ც', 'dz': 'ძ', 'ts': 'წ', 'ch': 'ჭ',
        'kh': 'ხ', 'j': 'ჯ', 'h': 'ჰ',
        # Common combinations
        'sh': 'შ', 'ch': 'ჩ', 'ts': 'ც', 'dz': 'ძ', 'kh': 'ხ', 'gh': 'ღ'
    }
    
    # Try to convert romanized text to Georgian
    converted_text = text
    
    # Replace common combinations first (longer patterns)
    for roman, georgian in sorted(roman_to_georgian.items(), key=lambda x: len(x[0]), reverse=True):
        converted_text = converted_text.replace(roman.lower(), georgian)
    
    # If we successfully converted a significant portion, return the converted text
    georgian_char_count = sum(1 for char in converted_text if is_georgian(char))
    if georgian_char_count > len(text) * 0.3:  # If more than 30% became Georgian
        return converted_text
    
    # Otherwise, return original text
    return text


def get_language_name(lang_code: str) -> str:
    """Get human-readable language name from language code."""
    language_names = {
        'ka': 'Georgian',
        'en': 'English',
        'ru': 'Russian',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'ar': 'Arabic',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'hi': 'Hindi',
        'unknown': 'Unknown'
    }
    
    return language_names.get(lang_code, lang_code)


def is_multilingual(text: str) -> bool:
    """Check if text contains multiple languages."""
    if not text:
        return False
    
    # Split into sentences/segments
    segments = re.split(r'[.!?]+', text)
    
    languages = set()
    for segment in segments:
        if len(segment.strip()) > 5:  # Only check segments with meaningful content
            lang = detect_language(segment.strip())
            if lang != "unknown":
                languages.add(lang)
    
    return len(languages) > 1


def clean_whatsapp_artifacts(text: str) -> str:
    """Remove WhatsApp-specific artifacts like media placeholders and timestamps."""
    if not text:
        return text
    
    # Remove timestamp + name + any '... omitted' segments
    text = re.sub(
        r"\s*\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]\s*[^:]{1,100}:\s*[^\n]*?omitted\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    
    # Remove orphan media/contact/document/location placeholders
    text = re.sub(
        r"\b(?:image|video|audio|sticker|gif|document|contact(?:\s+card)?|location|photo|picture|file)\s+omitted\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_common_artifacts(text: str) -> str:
    """Remove common artifacts that appear across all platforms."""
    if not text:
        return text
    
    # Remove "<This message was edited>" strings (common in messaging apps)
    text = re.sub(r'\s*<This message was edited>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<This message was edited\.>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<Message was edited>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<Message was edited\.>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<Edited>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<Edited\.>\s*', ' ', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_gmail_artifacts(text: str) -> str:
    """Additional Gmail-specific cleaning for signatures and artifacts."""
    if not text:
        return text
    
    # Remove Gmail mobile signatures
    text = re.sub(r'\s*Sent from my (iPhone|iPad|Android|mobile device)\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*Get Outlook for (iOS|Android)\s*$', '', text, flags=re.IGNORECASE)
    
    # Remove Gmail-specific patterns like "> > >" chains
    text = re.sub(r'\s*>\s*>\s*>\s*', ' ', text)
    
    # Remove common email signatures
    text = re.sub(r'\s*--\s*\n.*$', '', text, flags=re.DOTALL)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_text_comprehensive(text: str, platform: str = None) -> str:
    """Comprehensive text cleaning for all platforms."""
    if not text:
        return text
    
    # Apply common cleaning first
    text = clean_common_artifacts(text)
    
    # Apply platform-specific cleaning
    if platform == "whatsapp":
        text = clean_whatsapp_artifacts(text)
    elif platform == "gmail":
        text = clean_gmail_artifacts(text)
    
    # Final normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
