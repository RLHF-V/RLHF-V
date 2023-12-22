import os
import json
import tqdm
import base64
import pandas

from muffin.data.tsv_file_op import multimodal_img_tsv_writer

# TODO: You have to re-implement regarding your source data format
def parquet_data_stream_dpo(origin_dataset_name, parquet_data_folder, datasetname, file_list):
    for file in tqdm.tqdm(file_list):
        print(file)

        list_data = list(pandas.read_parquet(os.path.join(parquet_data_folder, file)).iterrows())
        for idx, value in list_data:
            img_buffer = base64.b64encode(value['BUFFER']).decode('utf-8')
            value_text_rejected = value['rejected']
            value_text_chosen = value['chosen']
            value_text_question = value['question']

            value_text = [value_text_question, value_text_chosen, value_text_rejected]
            text = base64.b64encode(json.dumps(list(value_text)).encode('utf-8')).decode('utf-8')
            img_path = value['IMAGE_ID']

            dataset_name = datasetname
            origin_dataset = origin_dataset_name
            origin_split = base64.b64encode(json.dumps(value['metainfo']).encode('utf-8')).decode('utf-8')
            origin_split_inner_idx = f'{idx}'

            print(dataset_name, origin_dataset, origin_split, origin_split_inner_idx, img_path)

            yield dataset_name, img_buffer, text, origin_dataset, origin_split, origin_split_inner_idx, img_path

def dpo_preference_convert():
    datasetname = 'dpo_preference'
    parquet_data_folder = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20230919'
    file_list = os.listdir(parquet_data_folder)
    file_list = [file for file in file_list if file.endswith(".parquet")]
    print("file nums:", len(file_list))
    output_name = datasetname

    # TODO: output data with be ./DATA_NAME-DATA_SIZE.tsv and ./DATA_NAME-DATA_SIZE.tsv.lineidx

    multimodal_img_tsv_writer(
        parquet_data_stream_dpo(origin_dataset_name='dpo/DPO_preference_20230919',
            parquet_data_folder=parquet_data_folder,
            datasetname=datasetname,
            file_list=file_list), f'{parquet_data_folder}/{output_name}')

if __name__ == '__main__':
    dpo_preference_convert()
