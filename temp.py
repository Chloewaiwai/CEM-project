from src.utils.constants import EMOJI_MAP as emoji_map

def get_emoji(emotion):
    #emoji= emoji_map.get(emotion_word)
    emoji= emoji_map.get(emotion)
    return emoji

emoji=get_emoji("kwug")
print(emoji)