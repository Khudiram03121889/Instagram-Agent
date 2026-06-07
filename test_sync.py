import re
def _tokenize_words(text: str): return re.findall(r"[A-Za-z0-9']+", text.lower())
STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'by', 'for', 'from', 'has', 'have', 'how', 'if', 'in', 'into', 'is', 'it', 'its', 'of', 'on', 'or', 'so', 'that', 'the', 'their', 'them', 'there', 'these', 'this', 'to', 'was', 'we', 'what', 'when', 'where', 'which', 'why', 'with', 'you', 'your'}
def _important_tokens(text: str, min_len: int = 4): return [t for t in _tokenize_words(text) if len(t) >= min_len and t not in STOPWORDS]

voice_text = 'A yawn can spread fast because seeing or hearing one yawn can nudge your own brain toward the same action.'
visual = 'Realistic high-quality microscopic-to-macro educational style in a natural indoor setting with soft, high-contrast lighting. A close-up of one person in a small group begins a yawn: mouth opening wide, head tilting back slightly, shoulders relaxing. The camera quickly cuts to the nearby faces of two other people who notice it and start to mirror the same mouth opening and head tilt. Keep the motion fast and literal, showing the yawn spreading from one person to the others through visible reaction and timing, with no text or symbols.'
visual_goal = 'Viewer understands that one yawn can trigger another person to yawn right away.'
sync_terms = ['yawn', 'spread']

visual_tokens = set(_important_tokens(f'{visual_goal} {visual}', min_len=3))
print('visual_tokens:', visual_tokens)
for term in sync_terms:
    term_tokens = set(_important_tokens(term, min_len=3))
    print('term:', term, 'term_tokens:', term_tokens, 'intersects visual:', term_tokens.intersection(visual_tokens))
