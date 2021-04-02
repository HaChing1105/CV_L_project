import numpy as np
import json
import os
import argparse
import text_helper as text_processing
import torch
from pytorch_transformers import BertModel, BertTokenizer
from collections import defaultdict
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):

    print('building vqa %s dataset' % image_set)
    if image_set in ['train2014', 'val2014']:
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)
    image_name_template = 'COCO_'+coco_set_name+'_%012d'

    dataset = [None]*len(questions)
    bert_dataset = [None]*len(questions)

    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name+'.jpg')
        question_str = q['question']
        question_tokens = text_processing.tokenize(question_str)

        # This is data for BERT model
        question_str = "[CLS] " + question_str + " [SEP]"
        bert_tokens = bert_tokenizer.tokenize(question_str)  # question_str --> sentence

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)

        bert_iminfo = dict(
                    image_name=image_name,
                    image_path=image_path,
                    question_id=question_id,
                    question_str=question_str,
                    question_tokens=bert_tokens)
        
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)

            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers

            bert_iminfo['all_answers'] = all_answers
            bert_iminfo['valid_answers'] = valid_answers
            
        dataset[n_q] = iminfo
        bert_dataset[n_q] = bert_iminfo
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return dataset, bert_dataset


def main(args):
    
    image_dir = args.input_dir+'/Resized_Images/%s/'
    annotation_file = args.input_dir+'/Annotations/v2_mscoco_%s_annotations.json'
    question_file = args.input_dir+'/Questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = args.output_dir+'/vocab_answers.txt'
    answer_dict = text_processing.VocabDict(vocab_answer_file)

    valid_answer_set = set(answer_dict.word_list)    
    
    train, bert_train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2014')
    valid, bert_val = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'val2014')
    test, bert_test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2015')
    # test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')

    """
    Here we limit the number of data (word2vec data)
    """
    np.save(args.output_dir+'/train.npy', np.array(train[:50000]))
    np.save(args.output_dir+'/valid.npy', np.array(valid[:5000]))
    # np.save(args.output_dir+'/train_valid.npy', np.array(train+valid))
    np.save(args.output_dir+'/test.npy', np.array(test[:2000]))
    # np.save(args.output_dir+'/test-dev.npy', np.array(test_dev))

    """
    Here is for BERT data 
    """
    np.save(args.output_dir + '/bert_train.npy', np.array(bert_train[:50000]))
    np.save(args.output_dir + '/bert_valid.npy', np.array(bert_val[:5000]))
    np.save(args.output_dir + '/bert_test.npy', np.array(bert_test[:2000]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='../datasets',
                        help='directory for outputs')
    
    args = parser.parse_args()

    main(args)
