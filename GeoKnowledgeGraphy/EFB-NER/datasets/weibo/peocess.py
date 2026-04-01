import json
import os

def convert_file_to_json(input_file, output_file):
    data_list = []
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 -> {input_file}")
        return

    print(f"正在处理: {input_file} ...")

    with open(input_file, 'r', encoding='utf-8') as f:
        # 逐行读取，去除首尾空白
        lines = [line.strip() for line in f.readlines()]

    sentence_chars = []
    sentence_tags = []
    
    for line in lines:
        # 如果是空行，表示一个句子结束
        if not line:
            if sentence_chars:
                process_sentence(sentence_chars, sentence_tags, data_list)
                sentence_chars = []
                sentence_tags = []
            continue
        
        # 核心逻辑：自动适配列数
        parts = line.split()
        if len(parts) >= 2:
            char = parts[0]     # 第一列是字
            tag = parts[-1]     # 最后一列是 NER 标签 (兼容 2列 或 3列格式)
            
            sentence_chars.append(char)
            sentence_tags.append(tag)
        # 如果只有一列（如只有字没有标），可以视情况跳过或标记为 O
        elif len(parts) == 1:
             sentence_chars.append(parts[0])
             sentence_tags.append("O")

    # 处理文件末尾可能存在的最后一个句子
    if sentence_chars:
        process_sentence(sentence_chars, sentence_tags, data_list)

    # 写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in data_list:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"成功生成: {output_file} (共 {len(data_list)} 条数据)")

def process_sentence(chars, tags, data_list):
    text = "".join(chars)
    labels = {}
    
    i = 0
    while i < len(tags):
        tag = tags[i]
        
        # 解析 BIO/BMES 标签
        if tag.startswith("B-") or tag.startswith("S-"):
            try:
                # --- 类型处理 ---
                full_type = tag.split("-")[1] 
                
                # 【可选】如果你想合并所有子类型（如 PER.NAM -> PER），请取消下面这行的注释：
                # full_type = full_type.split('.')[0] 

                entity_type = full_type
            except IndexError:
                i += 1
                continue

            start_index = i
            end_index = i
            
            # 如果是 B- (Begin)，寻找后续的 I- (Inside)
            if tag.startswith("B-"):
                j = i + 1
                while j < len(tags):
                    next_tag = tags[j]
                    # 匹配逻辑：必须是 I- 且类型一致 (允许后缀不同，如 B-PER.NAM 接 I-PER.NAM)
                    # 简单的判断：只要是 I- 开头且包含相同主类型即可，或者严格匹配字符串
                    if next_tag.startswith("I-") and full_type in next_tag: 
                        end_index = j
                        j += 1
                    else:
                        break
                i = end_index 
            
            # 提取文本 (闭区间索引)
            entity_text = text[start_index : end_index + 1]
            
            # 构建输出字典
            if entity_type not in labels:
                labels[entity_type] = {}
            
            if entity_text not in labels[entity_type]:
                labels[entity_type][entity_text] = []
            
            labels[entity_type][entity_text].append([start_index, end_index])
            
        i += 1
    
    if text:
        data_list.append({
            "text": text,
            "label": labels
        })

# --- 运行部分 ---
if __name__ == "__main__":
    
    convert_file_to_json(r"C:\Users\dingyihu\Desktop\data\weibo\weibo命名实体识别数据集\train.txt", r"C:\Users\dingyihu\Desktop\data\weibo\train.json")
    convert_file_to_json(r"C:\Users\dingyihu\Desktop\data\weibo\weibo命名实体识别数据集\dev.txt", r"C:\Users\dingyihu\Desktop\data\weibo\dev.json")
    convert_file_to_json(r"C:\Users\dingyihu\Desktop\data\weibo\weibo命名实体识别数据集\test.txt", r"C:\Users\dingyihu\Desktop\data\weibo\test.json")