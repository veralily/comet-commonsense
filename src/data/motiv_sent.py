import utils.utils as utils
import src.data.utils as data_utils
import src.data.config as cfg

import pandas
import json
import random
import math
import torch

from tqdm import tqdm


def map_name(name):
    if name == "train":
        return "trn"
    elif name == "test":
        return "tst"
    else:
        return "dev"


class DataLoader(object):
    def __init__(self, opt):
        self.data = {}
        self.data["train"] = {}
        self.data["dev"] = {}
        self.data["test"] = {}

        self.sequences = {}
        self.sequences["train"] = {}
        self.sequences["dev"] = {}
        self.sequences["test"] = {}

        self.masks = {}
        self.masks["train"] = {}
        self.masks["dev"] = {}
        self.masks["test"] = {}

        self.offsets = {}
        self.offsets["train"] = {}
        self.offsets["dev"] = {}
        self.offsets["test"] = {}

    def offset_summary(self, split):
        return self.offsets[split]["total"]


def do_take_partial_dataset(data_opts):
    if data_opts.get("kr", None) is None:
        return False
    if data_opts.kr == 1:
        return False
    return True


def select_partial_dataset(data_opts, data):
    num_selections = math.ceil(data_opts.kr * len(data))
    return random.sample(data, num_selections)


class GenerationDataLoader(DataLoader):
    def __init__(self, opt, categories):
        super(GenerationDataLoader, self).__init__(opt)

        self.categories = categories
        self.opt = opt

        for split in self.data:
            self.data[split] = {"total": []}
            self.offsets[split] = {"total": 0}

        self.vocab_encoder = None
        self.vocab_decoder = None
        self.special_chars = None
        self.max_event = None
        self.max_effect = None

    def load_data(self, path):
        if ".pickle" in path:
            print("Loading data from: {}".format(path))
            data_utils.load_existing_data_loader(self, path)

            return True

        for split in self.data:
            file_name = "motiv_sent_none_{}.csv".format(map_name(split))
            print(f"read file: {file_name}")

            df = pandas.read_csv("{}/{}".format(path, file_name))
            print(f"columns: {df.columns}")
            sentences = []
            for i in range(len(df.values)):
                context = df.loc[i, "context"].split('\t')[1:]
                linenum = int(df.loc[i, "linenum"])
                char = df.loc[i, "char"]
                sentences.append('|'.join(context[0:linenum+1])+f"</s>{char}<s>")
            targets = df["motivation"].values

            if len(self.categories) == 1:
                cat = self.categories[0]

                self.data[split]["total"] += [(str(sent), str(rel), str(tar)) for sent, rel, tar in zip(
                    sentences, ["<{}>".format(cat)] * len(sentences), targets)]

        if do_take_partial_dataset(self.opt.data):
            self.data["train"]["total"] = select_partial_dataset(
                self.opt.data, self.data["train"]["total"])

        return False

    def make_tensors(self, text_encoder, special,
                     splits=["train", "dev", "test"], test=False):
        self.vocab_encoder = text_encoder.encoder
        self.vocab_decoder = text_encoder.decoder
        self.special_chars = special

        sequences = {}
        for split in splits:
            sequences[split] = get_generation_sequences(
                self.opt, self.data, split, text_encoder, test)

            self.masks[split]["total"] = [(len(i[0]), len(i[1])) for
                                          i in sequences[split]]

        self.max_event = max([max([l[0] for l in self.masks[split]["total"]])
                              for split in self.masks])
        self.max_effect = max([max([l[1] for l in self.masks[split]["total"]])
                               for split in self.masks])

        print(self.max_event)
        print(self.max_effect)

        for split in splits:
            num_elements = len(sequences[split])
            self.sequences[split]["total"] = torch.LongTensor(
                num_elements, self.max_event + self.max_effect).fill_(0)

            for i, seq in enumerate(sequences[split]):
                # print(self.sequences[split]["total"][i, :len(seq[0])].size())
                # print(torch.FloatTensor(seq[0]).size())
                self.sequences[split]["total"][i, :len(seq[0])] = \
                    torch.LongTensor(seq[0])
                self.sequences[split]["total"][i, self.max_event:self.max_event + len(seq[1])] = \
                    torch.LongTensor(seq[1])

    def sample_batch(self, split, bs, idxs=None):
        offset = self.offsets[split]["total"]

        batch = {}

        # Decided not to reduce computation on here because it's all parallel
        # anyway and we don't want to run out of memory in cases where we
        # don't see the longest version quickly enough

        if idxs:
            seqs = self.sequences[split]["total"].index_select(
                0, torch.LongTensor(idxs).to(
                    self.sequences[split]["total"].device))
        else:
            seqs = self.sequences[split]["total"][offset:offset + bs]
        batch["sequences"] = seqs.to(cfg.device)
        batch["attention_mask"] = make_attention_mask(seqs)
        batch["loss_mask"] = make_loss_mask(
            seqs, self.max_event, 1)
        batch["key"] = ("total", offset, offset + bs)

        offset += seqs.size(0)

        self.offsets[split]["total"] = offset

        if split == "train" and offset + bs > len(self.sequences[split]["total"]):
            return batch, True
        elif offset >= len(self.sequences[split]["total"]):
            return batch, True
        else:
            return batch, False

    def reset_offsets(self, splits=["train", "test", "dev"],
                      shuffle=True, keys=None):
        if isinstance(splits, str):
            splits = [splits]

        for split in splits:
            if keys is None:
                keys = ["total"]

            for key in keys:
                self.offsets[split][key] = 0

            if shuffle:
                self.shuffle_sequences(split, keys)

    def shuffle_sequences(self, split="train", keys=None):
        if keys is None:
            # print(type(self.data))
            # print(type(self.data.keys()))
            keys = self.data[split].keys()

        for key in keys:
            idxs = list(range(len(self.data[split][key])))

            random.shuffle(idxs)

            self.sequences[split][key] = \
                self.sequences[split][key].index_select(
                    0, torch.LongTensor(idxs))

            temp = [self.data[split][key][i] for i in idxs]
            self.data[split][key] = temp
            temp = [self.masks[split][key][i] for i in idxs]
            self.masks[split][key] = temp


def prune_data_for_evaluation(data_loader, categories, split):
    indices = []
    for i, example in enumerate(data_loader.data[split]["total"]):
        if example[1] in categories:
            indices.append(i)

    data_loader.masks[split]["total"] = [data_loader.masks[split]["total"][i]
                                         for i in indices]
    data_loader.sequences[split]["total"] = \
        data_loader.sequences[split]["total"].index_select(
            0, torch.LongTensor(indices))
    data_loader.data[split]["total"] = [data_loader.data[split]["total"][i]
                                        for i in indices]


def make_attention_mask(sequences):
    return (sequences != 0).float().to(cfg.device)


def make_loss_mask(sequences, max_event, num_delim_tokens):
    # print(num_delim_tokens)
    # print(sequences.size())
    mask = (sequences != 0).float()
    mask[:, :max_event + num_delim_tokens] = 0
    return mask[:, 1:].to(cfg.device)


def find_underscore_length(seq):
    start = "_"

    while start in seq:
        start += "_"
    return start[:-1]


def handle_underscores(suffix, text_encoder, prefix=False):
    encoder = text_encoder.encoder
    if prefix:
        tok = "___"
    else:
        tok = find_underscore_length(suffix)

    suffix_parts = [i.strip() for i in suffix.split("{}".format(tok))]
    to_flatten = []
    for i, part in enumerate(suffix_parts):
        if part:
            to_flatten.append(text_encoder.encode([part], verbose=False)[0])

            if i != len(suffix_parts) - 1 and suffix_parts[i+1]:
                to_flatten.append([encoder["<blank>"]])
        else:
            to_flatten.append([encoder["<blank>"]])

    final_suffix = utils.flatten(to_flatten)

    return final_suffix


def get_generation_sequences(opt, data, split, text_encoder, test):
    sequences = []
    count = 0

    final_prefix = None
    final_suffix = None

    for prefix, category, suffix in tqdm(data[split]["total"]):
        final_prefix, final_suffix = do_example(
            text_encoder, prefix, suffix, True, True)
        # if do_prefix:
        #     if "___" in prefix:
        #         final_prefix = handle_underscores(prefix, text_encoder, True)
        #     else:
        #         final_prefix = text_encoder.encode([prefix], verbose=False)[0]
        # if do_suffix:
        #     if "_" in suffix:
        #         final_suffix = handle_underscores(suffix, text_encoder)
        #     else:
        #         final_suffix = text_encoder.encode([suffix], verbose=False)[0]

        final = compile_final_sequence(
            opt, final_prefix, final_suffix, category, text_encoder)

        sequences.append(final)

        count += 1

        if count > 10 and test:
            break

    return sequences



def do_example(text_encoder, prefix, suffix, do_prefix, do_suffix):
    final_prefix = None
    final_suffix = None

    if do_prefix:
        if "___" in prefix:
            final_prefix = handle_underscores(prefix, text_encoder, True)
        else:
            final_prefix = text_encoder.encode([prefix], verbose=False)[0]
    if do_suffix:
        if "_" in suffix:
            final_suffix = handle_underscores(suffix, text_encoder)
        else:
            final_suffix = text_encoder.encode([suffix], verbose=False)[0]

    return final_prefix, final_suffix


def compile_final_sequence(opt, final_prefix, final_suffix, category, text_encoder):
    final = []

    final.append(final_prefix)
    final.append(
        [text_encoder.encoder[category]]
        + final_suffix)

    final[-1].append(text_encoder.encoder["<END>"])

    return final


num_delimiter_tokens = {
    "category": 1,
    "hierarchy": 3,
    "hierarchy+label": 4,
    "category+hierarchy": 4,
    "category+hierarchy+label": 5
}
