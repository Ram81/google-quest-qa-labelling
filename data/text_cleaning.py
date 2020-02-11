import re
import html


def clean_urls(x):
    x = re.sub(r'http\S+', ' URL ', x)
    x = re.sub(r'www\S+', ' URL ', x)
    return re.sub(r'@\S+', ' USERNAME ', x)


def clean_apostrophes(x):
    apostrophes = ["’", "‘", "´", "`"]
    for s in apostrophes:
        x = re.sub(s, "'", x)
    return x


def clean_latex_tags(text):
    text = re.sub('(\[ math \]).+(\[ / math \])', 'MATHS', text)
    text = re.sub('(\$\$).+(\$\$)', 'MATHS', text)
    text = re.sub('(\$).+(\$)', 'MATHS', text)
    return text


spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', 
          '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0', '\t', '\n']

def clean_spaces(text):
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text


def clean_numbers(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    return text


def clean_text(text):
    # text = html.unescape(text)
    text = clean_apostrophes(text)
    text = clean_urls(text)
#     text = clean_numbers(text)
    text = clean_latex_tags(text)
    text = clean_spaces(text)
    
    return text