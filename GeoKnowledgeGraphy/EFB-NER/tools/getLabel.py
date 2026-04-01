import json
import os

def get_all_labels(file_path):
    """
    从 JSONL 数据集中提取所有唯一的实体标签类型。
    """
    # 使用集合 (set) 来存储标签，自动去重
    # 默认添加 'O'，因为非实体区域通常标记为 O，且不出现在 label 字典中
    label_set = set(['O'])
    
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                # data["label"] 是一个字典，如 {"LOC": {...}, "ORG": {...}}
                # 我们只需要提取它的键 (Keys)
                current_labels = data.get("label", {}).keys()
                label_set.update(current_labels)
            except json.JSONDecodeError:
                continue

    # 将集合转换为列表并排序
    sorted_labels = sorted(list(label_set))
    return sorted_labels

# --- 运行部分 ---
if __name__ == "__main__":
    # 请修改为您的文件名
    input_file = "/root/autodl-tmp/third/data/one/wb/train.json" 
    
    labels = get_all_labels(input_file)
    
    print(f"处理文件: {input_file}")
    print(f"共发现 {len(labels)} 种标签 (包含 'O'):")
    
    # 打印结果
    print(labels)
    
    # 如果你想把这个列表保存下来，可以取消下面这行的注释
    # with open("labels.txt", "w", encoding="utf-8") as f: f.write(json.dumps(labels))