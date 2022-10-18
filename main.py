import gc

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.loader import *
from src.model import *
import argparse
import logging
from transformers import BertTokenizer, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tqdm import tqdm, trange
import numpy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set parameters
    args = argparse.Namespace(dataset='bsz8_mixed_3sat_5var_mixed_10clauses_8_10k_HR',
                              batch_size=8,
                              max_seq_length=128,
                              train_size=0.8,
                              bert_version='bert-base-uncased',
                              random_seed=123,
                              lr=2e-5,
                              weight_decay=1e-3,
                              lr_schedule='linear',
                              epochs=30,
                              pooling='cls',
                              embedding='concat'
                              )

    model_path = os.path.join('finetuned_bert_base_128_cls', args.dataset + args.pooling + args.embedding + "_2mlp")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_params = ["bert-base-uncased", "_noCL"]

    model_params += [str(args.lr), str(args.batch_size), str(args.weight_decay), str(args.pooling), str(args.embedding)]

    logging_path = init_logging_path(model_path)
    print(logging_path)
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.INFO)
    logging.info(str(args))

    fname_main = "dataset/Sent_5var_HR/SENT_" + str(args.dataset) + "_main.csv"
    fname_allpairs = "dataset/Sent_5var_HR/SENT_" + str(args.dataset) + "_allpairs.csv"
    file_loader_path = os.path.join("src/dataset/", args.dataset)

    df_main = pd.read_csv(fname_main, sep=None, engine="python")
    print('df_main.shape: ', df_main.shape)
    # df_main: [id_set, clauses]

    bert_tokenizer = BertTokenizer.from_pretrained('libs/bert_tokenizer_base_uncased', local_files_only=True)

    df_allpairs = load_csv(fname_allpairs)
    # all_pairs: [id, id_set, sentence1, sentence2, is_equivalent, is_entailed]
    print('df_allpairs.shape: ', df_allpairs.shape)

    # 1. Build dataset for contrastive learning per batch_size, id_set, and mode.
    # Splits the main dataset into train:eval:test = 8:1:1
    logging.info("build datasets...")
    x = df_main
    x_train, x_rem = train_test_split(x, train_size=args.train_size, random_state=args.random_seed)
    x_val, x_test = train_test_split(x_rem, test_size=0.5)
    del x
    del x_rem
    msg = '(x_train, x_val, x_test): ' + str(len(x_train)) + ', ' + str(len(x_val)) + ', ' + str(len(x_test))
    logging.info(msg)
    # x_train, x_val, x_test have (id_set, clause)

    # 2. Extract sentences and labels
    df_train, df_val, df_test = load_pairs(x_train, x_val, x_test, file_loader_path, df_allpairs, args.batch_size)
    train_labels = df_train.is_entailed.values
    # df_train, df_val, df_test have (id, id_set, is_equivalent, is_entailed)

    # 3. Create the loaders
    train_loader = dataloader(df_train, args.batch_size)
    val_loader = dataloader(df_val, args.batch_size)
    test_loader = dataloader(df_test, args.batch_size)
    print('train_loader length: ', len(train_loader))
    del df_train
    del df_val
    del df_test

    # load sentences
    logging.info("load sentences ...")
    main_sentences = df_allpairs["sentence1"].tolist()
    logging.info("total number of main sentences: " + str(len(main_sentences)))
    pair_sentences = df_allpairs["sentence2"].tolist()
    logging.info("total number of pair sentences: " + str(len(pair_sentences)))
    cl_labels = df_allpairs["is_entailed"].tolist()
    logging.info("total number of labels: " + str(len(cl_labels)))

    # 4. Build model
    model = BertModel_FineTuned(args, 'libs/bert_model_base_uncased')

    tuned_parameters = [{'params': [param for name, param in model.named_parameters()]}]

    optimizer = AdamW(tuned_parameters, lr=args.lr)

    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")
    #  early_stopping = EarlyStopping(patience=20, verbose=False, path=model_file, delta=1e-10)
    early_stopping = EarlyStopping(patience=3, verbose=True, path=model_file, delta=1e-10)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2,
                                                int(len(train_loader) * args.epochs))

    # 5. GPU setting
    logging.info("Setting GPU...")
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple GPUs")
            model = torch.nn.DataParallel(model)
        else:
            logging.info("use single GPU")
        model.to("cuda")

    # 6. Start training
    model.train()
    logging.info("Start training...")
    torch.set_printoptions(threshold=10)
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            # idx1 = batch[0][:, 1]  -- get id_set, not needed
            indices = batch[0][:, 0]
            #  print(indices)
            sentences_1 = np.array(main_sentences)[indices - 1]
            sentences_2 = np.array(pair_sentences)[indices - 1]
            labels = np.array(cl_labels)[indices - 1]
            labels = labels.astype(numpy.float32)

            # sentences = np.core.defchararray.add(sentences_1, " ")
            # sentences = np.core.defchararray.add(sentences, sentences_2)

            seq_in1 = tokenize_mask(sentences_1, args.max_seq_length, bert_tokenizer)
            seq_in2 = tokenize_mask(sentences_2, args.max_seq_length, bert_tokenizer)

            token_ids1, input_masks1 = convert_tuple_to_tensor(seq_in1, use_gpu)
            token_ids2, input_masks2 = convert_tuple_to_tensor(seq_in2, use_gpu)

            optimizer.zero_grad()
            loss = model(token_ids1, input_masks1, token_ids2, input_masks2, labels)

            if pd.isna(loss):
                print(indices)
                break
            # print(f"loss: {loss}")
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            del seq_in1, seq_in2
            del token_ids1, token_ids2
            del input_masks1, input_masks2
            del loss
            gc.collect()

        print('train_loss: ', train_loss)
        print('num steps: ', str(step + 1))
        train_loss /= (step + 1)
        print('epoch: ', epoch, ' train_loss: {:,}'.format(train_loss))
        torch.cuda.empty_cache()

        # validation
        model.eval()
        val_loss = 0
        for step, batch in enumerate(tqdm(val_loader, desc="validation")):
            idx_eval = batch[0][:, 0]

            sent_eval_1 = np.array(main_sentences)[idx_eval - 1]
            sent_eval_2 = np.array(pair_sentences)[idx_eval - 1]
            val_labels = np.array(cl_labels)[idx_eval - 1]
            val_labels = val_labels.astype(numpy.float32)

            seq_ev1 = tokenize_mask(sent_eval_1, args.max_seq_length, bert_tokenizer)
            seq_ev2 = tokenize_mask(sent_eval_2, args.max_seq_length, bert_tokenizer)

            token_ids1, input_masks1 = convert_tuple_to_tensor(seq_ev1, use_gpu)
            token_ids2, input_masks2 = convert_tuple_to_tensor(seq_ev2, use_gpu)

            loss = model(token_ids1, input_masks1, token_ids2, input_masks2, val_labels)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            val_loss += loss.item()

            del seq_ev1, seq_ev2
            del token_ids1, token_ids2
            del input_masks1, input_masks2
            del loss
            gc.collect()

        print('val_loss: ', val_loss)
        print('num steps: ', str(step + 1))
        val_loss /= (step + 1)
        model.train()

        print('epoch: ', epoch + 1, ' val_loss: {:,}'.format(val_loss))
        logging.info("Epoch: %d | train loss: %.4f | dev loss: %.4f ", epoch + 1, train_loss, val_loss)

        torch.cuda.empty_cache()

        if epoch > 3:  # 20
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained.")
            break

    torch.cuda.empty_cache()









