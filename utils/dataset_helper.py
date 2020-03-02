import string, array, json, re, os, pickle
from numpy import array
from utils.Config import Config

def clean_sentences(lines: list) -> list:
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table 
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        if len(pair[0]) + len(pair[1]) > 200:
            continue
        for line in pair:

            # normalizing unicode characters
            # line = normalize('NFD', line).encode('ascii', 'ignore')
            # line = line.decode('UTF-8')

            # tokenize on white space
            line = line.split()

            # convert to lowercase
            line = [word.lower() for word in line]
            # removing punctuation
            line = [word.translate(table) for word in line]

            # removing non-printable chars form each token
            # line = [re_print.sub('', w) for w in line]

            # removing tokens with numbers
            line = [word for word in line if not word.isnumeric()]

            line.insert(0,'<start> ')
            line.append(' <end>')
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    print("CLEAN ===" + str(cleaned[0]) + "===")
    return array(cleaned)

def get_pairs_from_file(path: str) -> list: 
    pairs = []
    with open(path, "r", encoding="UTF-8") as json_file:
        mess_json = json.load(json_file)
    mess_chat_list = mess_json["chats"]["list"]
    print("Loaded JSON messages.")

    my_mess, their_mess = "", ""
    for chat in mess_chat_list:
        pair = []
        
        for mess in chat["messages"]:
            if mess["type"] != "message" or type(mess["text"]) == list:
                continue
            
            if mess["from"] == "Hubert Morawski":
                if their_mess != "":
                    my_mess = mess["text"]
                    pairs.append([their_mess, my_mess.lower()])
                    their_mess = ""
            else:
                their_mess = mess["text"]
    print("EXAMPLE PAIR ===" + str(pairs[0]) + "===")
    return pairs

def create_dataset():
    config = Config()
    data_path = os.path.abspath(config.telegram_export_path)
    if not os.path.exists(data_path):
        raise ValueError(f'File {data_path} do not exist.')

    pairs = get_pairs_from_file(data_path)
    cleaned = clean_sentences(pairs)

    with open("dataset.pkl", "wb") as dataset_file:
        pickle.dump(cleaned, dataset_file)
    print("Done. Saved as dataset.pkl. Messages: " + str(len(cleaned)))

def get_dataset(create_if_not_exist=False):
    if create_if_not_exist and not os.path.exists("dataset.pkl"):
        create_dataset()
    print("Found dataset.pkl. Loading...")
    with open("dataset.pkl", "rb") as dataset_file:
        raw_data = pickle.load(dataset_file)
    print(f"Loading completed. Messages: {len(raw_data)}")
    return raw_data