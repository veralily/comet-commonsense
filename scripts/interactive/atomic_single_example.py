import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--model_file", type=str, default="models/motiv_sent-generation/motiv_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-4")
    # greedy; beam-# where # is the beam size; topk-# where # is k

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("motiv_sent", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    sampling_algorithm = args.sampling_algorithm
    print(f"cfg.device: {cfg.device}")

    # while True:
    #     input_event = "help"
    #     category = "help"
    #     sampling_algorithm = args.sampling_algorithm
    #
    #     while input_event is None or input_event.lower() == "help":
    #         input_event = input("Give an event (e.g., PersonX went to the mall): ")
    #
    #         if input_event == "help":
    #             interactive.print_help(opt.dataset)
    #
    #     while category.lower() == "help":
    #         category = input("Give an effect type (type \"help\" for an explanation): ")
    #
    #         if category == "help":
    #             interactive.print_category_help(opt.dataset)
    #
    #     while sampling_algorithm.lower() == "help":
    #         sampling_algorithm = input("Give a sampling algorithm (type \"help\" for an explanation): ")
    #
    #         if sampling_algorithm == "help":
    #             interactive.print_sampling_help()
    #
    #     sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    #
    #     if category not in data_loader.categories:
    #         category = "all"

    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    category = 'Intent'

    filename = 'data/motiv_sent_none/motiv_sent_none_tst.csv'
    output_file = 'data/motiv_sent_none/test_case_motiv_sent_none_on_comet.csv'
    result = []
    data = pd.read_csv(filename)
    num_data = len(data.values)
    print_interval = max(1, num_data // 10)
    for i in range(num_data):
        context = str(data.loc[i, 'context']).split('\t')[1:]
        char = data.loc[i, 'char']
        linenum = int(data.loc[i, 'linenum'])
        input_event = '|'.join(context[0:linenum+1]) + f"</s>{char}<s>"
        # relation = data.loc[i, 'relation']
        relation = "Intent"
        if relation == 'Intent':
            outputs = interactive.get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, category)
            predicted_sent = outputs[category]['beams']
            result.append('\t'.join(predicted_sent))
        if (i + 1) % print_interval == 0:
            print(i)
    df = pd.DataFrame(result)
    df.columns = ["predicted_refs"]
    df.to_csv(output_file)
    print(f"Writing done!")





