from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
tokenizer = VoiceBpeTokenizer(vocab_file="./xtts_vocab/vocab.json")
text1 = "Привет"
text2 = "Пока"

ids1 = tokenizer.encode(text1, lang="ru")
ids2 = tokenizer.encode(text2, lang="ru")

print(f"IDs 1: {ids1}")
print(f"IDs 2: {ids2}")
assert ids1 != ids2, "Токенайзер выдает одинаковые ID для разного текста!"
print(f"Max ID: {max(ids1 + ids2)}, Vocab size: 256000")
# IDs 1: [259, 3454, 3544]
# IDs 2: [259, 3415, 3431]
# Max ID: 3544, Vocab size: 256000