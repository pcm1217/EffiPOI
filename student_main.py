import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color
from teacher import TeaRec
from student import StuRec
from data.dataset import StuRecDataset
from data.dataset import TeaRecDataset
from trainer import StuRecTrainer


def finetune(model_name, dataset, pretrained_file='', finetune_mode='', **kwargs):
    # configurations initialization
    student_props = [f'props/{model_name}.yaml', 'props/finetune.yaml']
    print(student_props)

    print("-------------------------")

    # configurations initialization
    student_config = Config(model=StuRec, dataset=dataset, config_file_list=student_props, config_dict=kwargs)
    init_seed(student_config['seed'], student_config['reproducibility'])
    # logger initialization
    init_logger(student_config)
    logger = getLogger()
    logger.info(student_config)

    # dataset filtering

    vq_dataset = StuRecDataset(student_config)
    #dataset = create_dataset(student_config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(student_config, vq_dataset)  

    # model loading and initialization
    model = StuRec(student_config, train_data.dataset).to(student_config['device'])
    model.pq_codes = model.pq_codes.to(student_config['device'])


    # Load teacher model
    path_state_dict = "/data/"
    state = torch.load(path_state_dict)
    state_dict = state['state_dict'] 
    teacher = TeaRec(student_config, train_data.dataset).to(student_config['device'])
    teacher.load_state_dict(state_dict, strict=False)

    # trainer loading and initialization
    trainer = StuRecTrainer(student_config, model, teacher=teacher)
    #trainer = StuRecTrainer(student_config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=student_config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=student_config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return student_config['model'], student_config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': student_config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='StuRec', help='model name')
    parser.add_argument('-d', type=str, default='NYC', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=str, default='', help='fine-tune mode')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.m, args.d, pretrained_file=args.p, finetune_mode=args.f)