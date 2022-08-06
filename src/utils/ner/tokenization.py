import re
import torch
from transformers import AutoTokenizer
from src.globals import tokenizer, debug_print, device

def align_to_bert_tokenization(sentences, labels):
    """
    Tokenizes words using the BERT auto tokenizer.
    Then realigns labels to the last token of each word.
    Not-last tokens are labelled with a <pad> tag.
    Returns the resulting tuple of the two dimensional lists (sentences, labels)
    """
    tokenized_sentences = []
    aligned_labels = []

    for s, l in zip(sentences, labels):
        tokenized_sentence = tokenizer.tokenize(' '.join(s)) 
        aligned_label = []
        current_word = ''
        i = 0
        for token in tokenized_sentence:
            current_word += re.sub(r'^##', '', token)
            s[i] = s[i].replace('\xad', '')
            
            assert token == '[UNK]' or s[i].startswith(current_word)

            if token == '[UNK]' or s[i] == current_word:
                current_word = ''
                aligned_label.append(l[i])
                i += 1
            else:
                aligned_label.append('<pad>')
        
        assert len(tokenized_sentence) == len(aligned_label)

        tokenized_sentences.append(tokenized_sentence)
        aligned_labels.append(aligned_label)
    
    return tokenized_sentences, aligned_labels


def convert_to_ids(sentences, taggings):
    """
    Transform tokens and string labels into tensors and int labels.
    Because non-entity words reserve the 0 int label, <pad> tags are mapped to 9.
    """
    sentences_ids = []
    taggings_ids = []
    for sentence, tagging in zip(sentences, taggings):
        sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['SEP'])).long()
        tagging_tensor = torch.tensor([9] + [int(tag) if tag != '<pad>' else 9 for tag in tagging] + [9]).long()

        sentences_ids.append(sentence_tensor.to(device))
        taggings_ids.append(tagging_tensor.to(device))
    return sentences_ids, taggings_ids

def tokenize(sentences, labels):
    bert_tokenized_sentences, aligned_taggings = align_to_bert_tokenization(sentences, labels)
    debug_print(sentences[0])
    debug_print(labels[0])
    debug_print(bert_tokenized_sentences[0])
    debug_print(aligned_taggings[0])
    sentences_ids, taggings_ids = convert_to_ids(bert_tokenized_sentences, aligned_taggings)
    debug_print(sentences_ids[0])
    debug_print(taggings_ids[0])
    return sentences_ids, taggings_ids