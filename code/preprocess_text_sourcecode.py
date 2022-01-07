


# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import metrics
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import re
import string
import codecs
from underthesea import word_tokenize
import re
from advertools.emoji import EMOJI

def normalize_text(text):
    '''
    Source code: https://github.com/tuanpham1989/sentiment_analysis_nal    
    '''
    
    text = str(text)
    
    #Từ điển tích cực, tiêu cực, phủ định
    path_nag = 'sentiment_dicts/neg.txt'
    path_pos = 'sentiment_dicts/pos.txt'
    path_not = 'sentiment_dicts/not.txt'

    with codecs.open(path_nag, 'r', encoding='UTF-8') as f:
        nag = f.readlines()
    nag_list = [n.replace('\n', '') for n in nag]

    with codecs.open(path_pos, 'r', encoding='UTF-8') as f:
        pos = f.readlines()
    pos_list = [n.replace('\n', '') for n in pos]
    with codecs.open(path_not, 'r', encoding='UTF-8') as f:
        not_ = f.readlines()
    not_list = [n.replace('\n', '') for n in not_]
    
    #Remove các ký tự kéo dài: vd: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # Chuyển thành chữ thường
    text = text.lower()

    #Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": "tiêu_cực", "👻": "tích_cực", "💃": "tích_cực",'🤙': ' tích_cực ', '👍': ' tích_cực ',
        "💄": "tích_cực", "💎": "tích_cực", "💩": "tích_cực","😕": "tiêu_cực", "😱": "tiêu_cực", "😸": "tích_cực",
        "😾": "tiêu_cực", "🚫": "tiêu_cực",  "🤬": "tiêu_cực","🧚": "tích_cực", "🧡": "tích_cực",'🐶':' tích_cực ',
        '👎': ' tiêu_cực ', '😣': ' tiêu_cực ','✨': ' tích_cực ', '❣': ' tích_cực ','☀': ' tích_cực ',
        '♥': ' tích_cực ', '🤩': ' tích_cực ', 'like': ' tích_cực ', '💌': ' tích_cực ',
        '🤣': ' tích_cực ', '🖤': ' tích_cực ', '🤤': ' tích_cực ', ':(': ' tiêu_cực ', '😢': ' tiêu_cực ',
        '❤': ' tích_cực ', '😍': ' tích_cực ', '😘': ' tích_cực ', '😪': ' tiêu_cực ', '😊': ' tích_cực ',
        '?': ' ? ', '😁': ' tích_cực ', '💖': ' tích_cực ', '😟': ' tiêu_cực ', '😭': ' tiêu_cực ',
        '💯': ' tích_cực ', '💗': ' tích_cực ', '♡': ' tích_cực ', '💜': ' tích_cực ', '🤗': ' tích_cực ',
        '^^': ' tích_cực ', '😨': ' tiêu_cực ', '☺': ' tích_cực ', '💋': ' tích_cực ', '👌': ' tích_cực ',
        '😖': ' tiêu_cực ', '😀': ' tích_cực ', ':((': ' tiêu_cực ', '😡': ' tiêu_cực ', '😠': ' tiêu_cực ',
        '😒': ' tiêu_cực ', '🙂': ' tích_cực ', '😏': ' tiêu_cực ', '😝': ' tích_cực ', '😄': ' tích_cực ',
        '😙': ' tích_cực ', '😤': ' tiêu_cực ', '😎': ' tích_cực ', '😆': ' tích_cực ', '💚': ' tích_cực ',
        '✌': ' tích_cực ', '💕': ' tích_cực ', '😞': ' tiêu_cực ', '😓': ' tiêu_cực ', '️🆗️': ' tích_cực ',
        '😉': ' tích_cực ', '😂': ' tích_cực ', ':v': '  tích_cực ', '=))': '  tích_cực ', '😋': ' tích_cực ',
        '💓': ' tích_cực ', '😐': ' tiêu_cực ', ':3': ' tích_cực ', '😫': ' tiêu_cực ', '😥': ' tiêu_cực ',
        '😃': ' tích_cực ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' tích_cực ', '🤝': ' tích_cực ', '🎈': ' tích_cực ',
        '😗': ' tích_cực ', '🤔': ' tiêu_cực ', '😑': ' tiêu_cực ', '🔥': ' tiêu_cực ', '🙏': ' tiêu_cực ',
        '🆗': ' tích_cực ', '😻': ' tích_cực ', '💙': ' tích_cực ', '💟': ' tích_cực ',
        '😚': ' tích_cực ', '❌': ' tiêu_cực ', '👏': ' tích_cực ', ';)': ' tích_cực ', '<3': ' tích_cực ',
        '🌝': ' tích_cực ',  '🌷': ' tích_cực ', '🌸': ' tích_cực ', '🌺': ' tích_cực ',
        '🌼': ' tích_cực ', '🍓': ' tích_cực ', '🐅': ' tích_cực ', '🐾': ' tích_cực ', '👉': ' tích_cực ',
        '💐': ' tích_cực ', '💞': ' tích_cực ', '💥': ' tích_cực ', '💪': ' tích_cực ',
        '💰': ' tích_cực ',  '😇': ' tích_cực ', '😛': ' tích_cực ', '😜': ' tích_cực ',
        '🙃': ' tích_cực ', '🤑': ' tích_cực ', '🤪': ' tích_cực ','☹': ' tiêu_cực ',  '💀': ' tiêu_cực ',
        '😔': ' tiêu_cực ', '😧': ' tiêu_cực ', '😩': ' tiêu_cực ', '😰': ' tiêu_cực ', '😳': ' tiêu_cực ',
        '😵': ' tiêu_cực ', '😶': ' tiêu_cực ', '🙁': ' tiêu_cực ',
        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '  tích_cực ', ':)': ' tích_cực ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' tích_cực ','hehe': ' tích_cực ','hihi': ' tích_cực ', 'haha': ' tích_cực ', 'hjhj': ' tích_cực ',
        'cute': u' dễ thương ','huhu': ' tiêu_cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích_cực ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích_cực ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback '}


    for k, v in replace_list.items():
        text = text.replace(k, v)

    # chuyen punctuation thành space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    text = word_tokenize(text, format = "text")
    texts = text.split()
    len_text = len(texts)
    texts = [t.replace('_', ' ') for t in texts]
    for i in range(len_text):
        cp_text = texts[i]
        if cp_text in not_list: # Xử lý vấn đề phủ định (VD: áo này chẳng đẹp--> áo này tiêu_cực)
            numb_word = 2 if len_text - i - 1 >= 4 else len_text - i - 1

            for j in range(numb_word):
                if texts[i + j + 1] in pos_list:
                    texts[i] = 'tiêu_cực'
                    texts[i + j + 1] = ''

                if texts[i + j + 1] in nag_list:
                    texts[i] = 'tích_cực'
                    texts[i + j + 1] = ''
                    
        
        else: #Thêm feature cho những sentiment words (áo này đẹp--> áo này đẹp tích_cực)
            if cp_text in pos_list:
                texts.append('tích_cực')
            elif cp_text in nag_list:
                texts.append('tiêu_cực')
        
    
    text = u' '.join(texts)

    #remove nốt những ký tự thừa thãi
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻','')
    return text

def pre_processing(text):
    
    text = str(text)
    
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # remove numbers
    text = re.sub("\d+[a-zA-Z]*", "", text)
    
    # remove hashtags
    text = re.sub("[#][^\s]+", "", text)
    
    # standardize text + tiêu_cực/tích_cực replacements
    text = normalize_text(text)
    
    # remove remaining emojis not classified as tích_cực or tiêu_cực
    EMOJI_PATTERN = EMOJI
    text = re.sub(EMOJI_PATTERN, r'', text)
    
    return text

def tokenize(text):
    return word_tokenize(text, format = "text")