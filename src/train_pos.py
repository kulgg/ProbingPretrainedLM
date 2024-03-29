from datasets import load_dataset
from src.models.nn.bertprobe import LinearProbeBert
from src.models.nn.linearbert import LinearBert
from src.models.nn.multilayerprobe import MultilayerProbeBert
from src.models.nn.randomprobe import LinearProbeRandom
from src.models.nn.resettedbert import ProbeResettedBert
from src.utils.pos.dataset_loader import load
from src.utils.helper import GetTotalWordCount
from src.utils.pos.tokenization import tokenize
from src.utils.shared.dataloader import data_loader
from src.utils.pos.test import test
from src.utils.pos.train import fit
from src.globals import *
from src.models.models_enum import Models

def go(params):
    """
    Main entry point for POS training.
    First loads the datasets and runs tokenization.
    Then executes training for the model specified in params.
    Finally the model is evaluated using the test dataset.
    """
    train_sentences, train_labels = load(TRAIN_FILE)
    eval_sentences, eval_labels = load(EVAL_FILE)
    test_sentences, test_labels = load(TEST_FILE)
    print("Training dataset sentences", len(train_sentences))
    print("Training dataset total words", GetTotalWordCount(train_sentences))
    print("Eval dataset sentences", len(eval_sentences))
    print("Eval dataset total words", GetTotalWordCount(eval_sentences))
    print("Test dataset sentences", len(test_sentences))
    print("Test dataset total words", GetTotalWordCount(test_sentences))

    label_vocab['<pad>'] = 0

    train_sentences_ids, train_tagging_ids = tokenize(train_sentences, train_labels)
    eval_sentences_ids, eval_tagging_ids = tokenize(eval_sentences, eval_labels)
    test_sentences_ids, test_tagging_ids = tokenize(test_sentences, test_labels)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids, params.batch_size)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids, params.batch_size)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids, params.batch_size)


    if params.model == Models.LPB.value:
        bert_probe_model = LinearProbeBert(len(label_vocab))
        print("TRAINING BERT PROBE")
        fit(bert_probe_model, train_loader, eval_loader, params)
        random_loss, random_accuracy = test(bert_probe_model, test_loader)
        print(f"Bert Probe Test: Loss {random_loss} Accuracy {random_accuracy}")
    elif params.model == Models.LPR.value:
        linear_model = LinearProbeRandom(len(label_vocab))
        print("TRAINING LINEAR RANDOM")
        fit(linear_model, train_loader, eval_loader, params)
        random_loss, random_accuracy = test(linear_model, test_loader)
        print(f"Random Test: Loss {random_loss} Accuracy {random_accuracy}")
    elif params.model == Models.LB.value:
        linear_bert_model = LinearBert(len(label_vocab))
        print("TRAINING BERT LINEAR")
        fit(linear_bert_model, train_loader, eval_loader, params)
        random_loss, random_accuracy = test(linear_bert_model, test_loader)
        print(f"Bert Linear Test: Loss {random_loss} Accuracy {random_accuracy}")
    elif params.model == Models.LPRB.value:
        resetted_bert = ProbeResettedBert(len(label_vocab))
        print("TRAINING RESETTED BERT PROBE")
        fit(resetted_bert, train_loader, eval_loader, params)
        random_loss, random_accuracy = test(resetted_bert, test_loader)
        print(f"Resetted Bert Test: Loss {random_loss} Accuracy {random_accuracy}")
    elif params.model == Models.MPB.value:
        multilayer_probe_bert = MultilayerProbeBert(len(label_vocab))
        print("TRAINING MULTILAYER BERT PROBE")
        fit(multilayer_probe_bert, train_loader, eval_loader, params)
        random_loss, random_accuracy = test(multilayer_probe_bert, test_loader)
        print(f"Multilayer Probe Bert Test: Loss {random_loss} Accuracy {random_accuracy}")