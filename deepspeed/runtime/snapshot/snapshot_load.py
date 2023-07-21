import torch


def load_file(filename):
    data = torch.load(filename)
    print(data)


if __name__ == "__main__":
    filename = "/home/ec2-user/zhuang/DeepSpeedExamples/bing_bert/saved_models/bing_bert_base_seq/snapshot/0.pt"
    load_file(filename)