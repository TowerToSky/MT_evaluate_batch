from comet import download_model, load_from_checkpoint
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import json
import shutil
# os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[1]
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp.unite.configuration import InputFormat
from bert_score import BERTScorer

lang_file = {
    'deu':'deu_Latn',
    'rus':'rus_Cyrl',
    'spa':'spa_Latn',
    'zho':'zho_Hans',
    'eng':'eng_Latn'
}

single_file = {
    'deu':'de',
    'rus':'ru',
    'spa':'es',
    'zho':'zh',
    'eng':'en',
    'isl':'is',
    'fra':'fr',
    'de':'de',
    'ru':'ru',
    'es':'es',
    'zh':'zh',
    'en':'en',
    'is':'is',
    'ha':'ha',
    'cs':'cs',
    'ja':'ja',
    'fr':'fr',
    'uk':'uk'
}

# 选择你所想要使用的指标
# used_metrics = ["sacre-bleu", "comet", "unite", "bert-score"]
used_metrics = ["sacre-bleu" ,"comet"]

# 翻译的方向
transalte_direct = [
    "cs->en",
    "en->cs",
    "de->en",
    "en->de",
    "de->fr",
    "fr->de",
    "uk->en",
    "en->uk",
    "zh->en",
    "en->zh",
]
d = {}
um = {}
for i in used_metrics:
    um[i] = 0.0
for td in transalte_direct:
    d[td] = um.copy()


def list_average(input_list):
    total = 0.0
    for i in input_list:
        total += i
    return total / len(input_list)


def unite_caculate(src_list, ref_list, tgt_list, pipeline_ins):
    unite_list = []
    if len(src_list) == len(ref_list) == len(tgt_list):
        for i in tqdm(range(len(src_list)), f"unite:"):
            cur_input = {
                "hyp": [src_list[i]],
                "src": [ref_list[i]],
                "ref": [tgt_list[i]],
            }
            unite_list.append(pipeline_ins(cur_input)["score"][0])
    return list_average(unite_list)


def batch_score_flores(flores_path, src_path, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_num
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    file_name = []
    score_list = []
    bleu_list = []
    total_ = len(os.listdir(src_path))
    # print(f'Total: {total_}')
    count_ = 1
    for file in os.listdir(src_path):
        data = []
        src_lang = file.split('.')[0].split('+')[0]
        tgt_lang = file.split('.')[0].split('2')[-1]
        mt_list = []
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as reader:
            for line in reader:
                mt_list.append(line)
        src_list = []
        with open(os.path.join(flores_path, lang_file[src_lang]+'.devtest'), 'r', encoding='utf-8') as reader:
            for line in reader:
                src_list.append(line)
        tgt_list = []
        with open(os.path.join(flores_path, lang_file[tgt_lang]+'.devtest'), 'r', encoding='utf-8') as reader:
            for line in reader:
                tgt_list.append(line)
        for i in range(len(mt_list)):
            data.append({
                "src":src_list[i],
                "mt":mt_list[i],
                "ref":tgt_list[i]
            })
        print(f'Processing {count_}/{total_}')
        model_output = model.predict(data, batch_size=64, gpus=1)
        file_name.append(file.split('.')[0])
        score_list.append(model_output[1])
        src = os.path.join(src_path, file)
        ref = os.path.join(flores_path, lang_file[tgt_lang]+'.devtest')
        command = f"cat {src} | sacrebleu {ref} -l {single_file[src_lang]}-{single_file[tgt_lang]} -m bleu chrf --chrf-word-order 2 -f text"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        bleu_list.append(str(result))
        # bleu_list.append(str(result).split('BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1 = ')[-1].split('/')[0].split(' ')[0])
        count_ += 1
    
    
    result_data = {'file': file_name, 'bleu':bleu_list, 'comet22-score': score_list}
    excel_filename = f'comet_score{src_path}.xlsx'
    df = pd.DataFrame(result_data)
    df.to_excel(excel_filename, index=False)

def batch_score_wmt(wmt_path, src_path, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_num
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)    # 但是每次加载模型好慢，所以两个模型能放下就放的下吧
    file_name = []
    score_list = []
    bleu_list = []
    score_list = []
    score_list_unite = []
    total_ = len(os.listdir(src_path))
    print(f'Total: {total_}')
    count_ = 1
    for file in os.listdir(src_path):
        temp = file
        data = []
        # src_lang = single_file[file.replace('GPT4-', '').split('-')[1].split('2')[0].split('_')[0]]
        # tgt_lang = single_file[file.split('2')[-1].split('-')[0]]
        src_lang = file.split('$')[-1].split('_')[0].replace('translation','')[:2]
        tgt_lang = file.split('$')[-1].split('_')[0].replace('translation','')[2:]

        mt_list = []
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as reader:
            for line in reader:
                mt_list.append(line)
        src_list = []
        with open(os.path.join(wmt_path, src_lang+tgt_lang, f'test.{src_lang}-{tgt_lang}.{src_lang}'), 'r', encoding='utf-8') as reader:
            for line in reader:
                src_list.append(line)
        tgt_list = []
        with open(os.path.join(wmt_path, src_lang+tgt_lang, f'test.{src_lang}-{tgt_lang}.{tgt_lang}'), 'r', encoding='utf-8') as reader:
            for line in reader:
                tgt_list.append(line)
        for i in range(len(mt_list)):
            data.append({
                "src":src_list[i],
                "mt":mt_list[i],
                "ref":tgt_list[i]
            })
        print(f'Processing {count_}/{total_}')
        print("## using comet22...")
        model_output = model.predict(data, batch_size=32, gpus=1)
        print("## using unite...")
        file_name.append(file.replace(".txt", "").replace(".out", ""))
        score_list.append(model_output[1])

        file_name.append(file.split('.')[0])
        score_list.append(model_output[1])
        src = os.path.join(src_path, temp).replace('$','\$')
        ref = os.path.join(wmt_path, src_lang+tgt_lang, f'test.{src_lang}-{tgt_lang}.{tgt_lang}')
        command = f"cat {src} | sacrebleu {ref} -l {single_file[src_lang]}-{single_file[tgt_lang]} -m bleu chrf --chrf-word-order 2 -f text"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        bleu_list.append(str(result).split('BLEU')[-1].split(' (BP')[0].split('2.4.0 = ')[-1].split(' ')[0])
        print(str(result))
        count_ += 1

    # result_data = {'file': file_name, 'bleu':bleu_list, 'comet22-score': score_list}
    # excel_filename = f'comet_score{gpu_num}.xlsx'
    # df = pd.DataFrame(result_data)
    # df.to_excel(excel_filename, index=False)

def extract_file(data_path, out_path):
    for file in tqdm(os.listdir(data_path)):
        with open(os.path.join(out_path, file+'.txt'), 'w', encoding='utf-8') as writer:
            with open(os.path.join(data_path, file, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as reader:
                for line in reader:
                    writer.write(json.loads(line)['predict'].replace('\n','') + '\n')

def batch_score_detect_trans(src_path):
    # os.environ['CUDA_VISIBLE_DEVICES']=gpu_num
    # model_path = download_model("Unbabel/wmt22-comet-da")
    # model = load_from_checkpoint(model_path)
    file_name = []
    bleu_list = []
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    score_list = []
    total_ = len(os.listdir(src_path))
    # print(f'Total: {total_}')
    count_ = 1
    for file in os.listdir(src_path):
        print(f'################### {file} ###################')
        src = file.split('_')[-2].split('-')[0].split('2')[0]
        tgt = file.split('_')[-2].split('-')[0].split('2')[-1]
        # print(f'{src}2{tgt}')
        # continue
        data = []
        src_list = []
        with open(os.path.join(src_path, file, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as reader:
            for line in reader:
                src_list.append(json.loads(line)['predict'])
        # 临时存储
        with open(f'{file}.out', 'w', encoding='utf-8') as writer:
            for line in src_list:
                writer.write(line.replace('\n', '')+'\n')
        tgt_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[tgt]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                tgt_list.append(line)
        ref_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[src]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                ref_list.append(line)
        for i in range(len(src_list)):
            data.append({
                "src":ref_list[i],
                "mt":src_list[i],
                "ref":tgt_list[i]
            })
        print(f'Processing {count_}/{total_}')
        # model_output = model.predict(data, batch_size=512, gpus=1)
        file_name.append(file.split('.')[0])
        # score_list.append(model_output[1])
        src_ = src
        # src = os.path.join(src_path, file)
        src = f'{file}.out'
        ref = f'datasets/wmt-testset/{single_file[src_]}{single_file[tgt]}/test.{single_file[src_]}-{single_file[tgt]}.{single_file[tgt]}'
        command = f"cat {src} | sacrebleu {ref} -l en-zh -m bleu chrf --chrf-word-order 2 -f text"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        print(result)
        os.remove(f'{file}.out')
        bleu_list.append(str(result).split('BLEU|nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.4.2 = ')[-1].split(' (BP = ')[0])
        count_ += 1
    
    # result_data = {'file': file_name, 'comet':score_list, 'sacre-bleu':bleu_list, 'bleu1':bleu1_list, 'bleu2':bleu2_list, 'bleu1':bleu1_list, 'bleu3':bleu3_list, 'bleu4':bleu4_list}
    result_data = {'file': file_name, 'comet':score_list, 'sacre-bleu':bleu_list}
    temp_name = src_path.split('/')[-1]
    excel_filename = f'{temp_name}_out.xlsx'
    df = pd.DataFrame(result_data)
    df.to_excel(excel_filename, index=False)

def batch_score_detect_file(src_path):
    # os.environ['CUDA_VISIBLE_DEVICES']=gpu_num

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    file_name = []
    bleu_list = []
    score_list = []
    total_ = len(os.listdir(src_path))
    count_ = 1
    for file in os.listdir(src_path):
        print(f'###################### now is {file} ######################')
        if 'GPT4' in file:
            file_temp = file.replace('wmt22-', '').replace('wmt22-GPT4-', '')
            src = file_temp.split('2')[0].split('_')[0]
            tgt = file_temp.split('2')[-1].split('-')[0]
        else:
            file_temp = file.replace('wmt22-', '').replace('wmt22-GPT4-', '')
            src = file_temp.split('2')[0].split('-')[0]
            tgt = file_temp.split('2')[-1].split('-')[0]
        src = src.replace('GPT4-', '')
        data = []
        src_list = []
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as reader:
            for line in reader:
                src_list.append(line)
        tgt_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[tgt]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                tgt_list.append(line)
        ref_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[src]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                ref_list.append(line)
        for i in range(len(src_list)):
            data.append({
                "src":ref_list[i],
                "mt":src_list[i],
                "ref":tgt_list[i]
            })
        print(f'Processing {count_}/{total_}')
        model_output = model.predict(data, batch_size=256, gpus=1)
        file_name.append(file.split('.')[0])
        score_list.append(model_output[1])
        src_ = src
        src = os.path.join(src_path, file)
        ref = f'datasets/wmt-testset/{single_file[src_]}{single_file[tgt]}/test.{single_file[src_]}-{single_file[tgt]}.{single_file[tgt]}'
        command = f"cat {src} | sacrebleu {ref} -l en-zh -m bleu chrf --chrf-word-order 2 -f text"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        # print(result)
        bleu_list.append(str(result).split('BLEU|nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.4.2 = ')[-1].split(' (BP = ')[0].split(' ')[0])
        count_ += 1
    
    # result_data = {'file': file_name, 'comet':score_list, 'sacre-bleu':bleu_list, 'bleu1':bleu1_list, 'bleu2':bleu2_list, 'bleu1':bleu1_list, 'bleu3':bleu3_list, 'bleu4':bleu4_list}
    
    result_data = {'file': file_name, 'comet':score_list, 'sacre-bleu':bleu_list}
    # result_data = {'file': file_name, 'sacre-bleu':bleu_list}
    output_name = src_path.split('/')[-1]
    excel_filename = f'{output_name}_out.xlsx'
    df = pd.DataFrame(result_data)
    df.to_excel(excel_filename, index=False)


def batch_score_nllb(src_path):
    # os.environ['CUDA_VISIBLE_DEVICES']=gpu_num

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    # pipeline_ins = pipeline(
    #     task=Tasks.translation_evaluation,
    #     model="damo/nlp_unite_mup_translation_evaluation_multilingual_large",
    # )
    file_name = []
    score_list = []
    bleu_list = []
    score_list = []
    score_list_unite = []
    bert_score_list = []

    # 加载三种语言的bert-score模型
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    batch_size = 64
    en_model = BERTScorer(lang="en", rescale_with_baseline=True, batch_size=batch_size)
    zh_model = BERTScorer(lang="zh", rescale_with_baseline=True, batch_size=batch_size)
    other_model = BERTScorer(lang='de', rescale_with_baseline=True, batch_size=batch_size)

    total_ = len(os.listdir(src_path))
    count_ = 1

    # 修改一下
    # total_data = []
    for file in os.listdir(src_path):
        print(f'###################### now is {file} ######################')
        src = file.split('-')[-2].split('.')[0].split('2')[0]
        tgt = file.split('-')[-2].split('.')[0].split('2')[-1]
        # src = file.split('_')[-1].split('.')[0][:2]
        # tgt = file.split('_')[-1].split('.')[0][-2:]
        print(f'NOW IS {src} -> {tgt}')
        data = []
        src_list = []
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as reader:
            for line in reader:
                src_list.append(line)
        tgt_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[tgt]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                tgt_list.append(line)
        ref_list = []
        with open(f'datasets/wmt-testset/{single_file[src]}{single_file[tgt]}/test.{single_file[src]}-{single_file[tgt]}.{single_file[src]}', 'r', encoding='utf-8') as reader:
            for line in reader:
                ref_list.append(line)
        for i in range(len(src_list)):
            data.append({
                "src":ref_list[i],
                "mt":src_list[i],
                "ref":tgt_list[i]
            })

        print(f'Processing [{count_}/{total_}]')
        print("## using comet22...")
        model_output = model.predict(data, batch_size=32, gpus=1)
        # score_list_unite.append(
        #     unite_caculate(src_list, ref_list, tgt_list, pipeline_ins)
        # )
        file_name.append(file.replace('.txt', '').replace('.out', ''))
        score_list.append(model_output[1])
        src_ = src
        src = os.path.join(src_path, file)
        ref = f'datasets/wmt-testset/{single_file[src_]}{single_file[tgt]}/test.{single_file[src_]}-{single_file[tgt]}.{single_file[tgt]}'
        command = f"cat {src} | sacrebleu {ref} -l en-zh -m bleu chrf --chrf-word-order 2 -f text"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        # print(result)
        bleu_list.append(str(result).split('BLEU|nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.4.2 = ')[-1].split(' (BP = ')[0].split(' ')[0])

        # 加一下bert_score
        print("## using bert-score...")
        if tgt == 'en':
            P, R, F1 = en_model.score(src_list, tgt_list, verbose=True)
        elif tgt == 'zh':
            P, R, F1 = zh_model.score(src_list, tgt_list, verbose=True)
        else:
            P, R, F1 = other_model.score(src_list, tgt_list, verbose=True)
        # P, R, F1 = bert_score.score(src_list, tgt_list, lang=tgt, verbose=True, batch_size=64)
        bert_score_list.append(F1.mean().item())

        count_ += 1

    # result_data = {'file': file_name, 'comet':score_list, 'sacre-bleu':bleu_list, 'bleu1':bleu1_list, 'bleu2':bleu2_list, 'bleu1':bleu1_list, 'bleu3':bleu3_list, 'bleu4':bleu4_list}

    result_data = {
        "file": file_name,
        "comet": score_list,
        "sacre-bleu": bleu_list,
        "bert-score": bert_score_list,
    }
    # result_data = {'file': file_name, 'sacre-bleu':bleu_list}
    output_name = src_path.split('/')[-1]
    os.makedirs('excelfiles', exist_ok=True)
    excel_filename = f'excelfiles/{output_name}_out.xlsx'
    df = pd.DataFrame(result_data)
    df.to_excel(excel_filename, index=False)
    return output_name

def process_llf_data(src_path, temp_path):
    os.makedirs(temp_path, exist_ok=True)
    for file in tqdm(os.listdir(src_path), desc='exacting file...'):
        if len(os.listdir(os.path.join(src_path, file))) == 0:
            continue
        with open(os.path.join(src_path, file, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as reader:
            out_file = file.split('_')[0] + '-' + file.split('_')[-2].split('-')[0] + '-epoch' + file.split('_')[-1] + '.txt'
            with open(f'{temp_path}/{out_file}', 'w', encoding='utf-8') as writer:
                for line in reader:
                    temp_data = json.loads(line)['predict'].replace('\n', '').replace('\r', '')
                    writer.write(temp_data+'\n')

def out2res(excel_name):
    df = pd.read_excel(f"excelfiles/{excel_name}_out.xlsx")
    # 填充字典 models_scores
    for index, row in df.iterrows():
        file = row["file"]

        # 拆分文件名
        split = file.split("-")
        modelName = "-".join(split[:-2])
        lang = split[-2]
        lang = "->".join(lang.split(str(2)))
        models_scores = {}
        # 创建模型字典如果不存在
        if modelName not in models_scores:
            models_scores[modelName] = d
        # 更新模型字典

        for um in used_metrics:
            models_scores[modelName][lang][um] = row[um]

    # 打印结果
    for model, scores in models_scores.items():
        print(f"Model: {model}")
        for lang, metrics in scores.items():
            print(f"  {lang}: {metrics}")

    rows = []
    index = []

    for model, scores in models_scores.items():
        row = {}
        for lang, metrics in scores.items():
            for um in used_metrics:
                row[f"{lang} {um}"] = metrics[um]

        rows.append(row)
        index.append(model)

    out = pd.DataFrame(rows, index=index)
    print(out)
    # 保存到 Excel 文件
    out.to_excel(f"excelfiles/{excel_name}_res.xlsx", sheet_name="Scores")
    print(f"输出结果保存到excel：excelfiles/{excel_name}_res.xlsx")

def llm_mt_score(src_path):
    file_name = src_path.split('/')[-1]
    process_llf_data(src_path, file_name)
    output_name = batch_score_nllb(file_name)
    out2res(output_name)
    shutil.rmtree(file_name)
    print('done')

if __name__ == '__main__':
    # batch_score_detect_file('exam_out/final_remain')
    # llm_mt_score("exam_out/qwen1.5-1.8B-Chat-new_wmt22")
    llm_mt_score("exam_out/qwen1.5-1.8B-Chat-chaos_cp-1560_wmt22")

    # llm_mt_score("exam_out/qwen1.5-4B-Chat-1560_wmt22")
    # out2res("qwen1.5-1.8B-Chat-new_wmt22")
    # llm_mt_score('exam_out/baichuan2')
    # batch_score_nllb('exam_out/alam_lora_7b_wmt22')
