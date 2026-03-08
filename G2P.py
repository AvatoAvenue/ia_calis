import nltk
import re
from g2p_en import G2p

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')

g2p = G2p()

arpabet_to_ipa = {
"AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ",
"B":"b","CH":"tʃ","D":"d","DH":"ð","EH":"ɛ","ER":"ɝ",
"EY":"eɪ","F":"f","G":"ɡ","HH":"h","IH":"ɪ","IY":"i",
"JH":"dʒ","K":"k","L":"l","M":"m","N":"n","NG":"ŋ",
"OW":"oʊ","OY":"ɔɪ","P":"p","R":"r","S":"s","SH":"ʃ",
"T":"t","TH":"θ","UH":"ʊ","UW":"u","V":"v","W":"w",
"Y":"j","Z":"z","ZH":"ʒ"
}


def convert_to_ipa(text):

    phonemes = g2p(text)

    ipa = []

    for p in phonemes:

        if p == " ":
            ipa.append(" ")
            continue

        p = re.sub(r'\d', '', p)

        if p in arpabet_to_ipa:
            ipa.append(arpabet_to_ipa[p])

    return "".join(ipa)

text = "database query"

print(convert_to_ipa(text))