from torch.utils.data import Dataset, DataLoader
from text2vec import build_word2vec
import jieba


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloder(text_path):
    sentences = []
    labels = []
    print('load text')
    with open(text_path, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i%1000 == 0:
                print(i, '/', len(lines))
            text, label = line.strip().split('\t')
            seg_list = jieba.cut(text, cut_all=False)
            sentence = []
            for seg in seg_list:
                sentence.append(seg)
            sentences.append(sentence)
            labels.append(int(label))

    print('split train val set')
    X_train, X_test, y_train, y_test = build_word2vec(sentences, labels)

    print('load text to loader')
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, labels