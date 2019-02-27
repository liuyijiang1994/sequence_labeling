PAD = 0
BOS = 1
EOS = 2
UNK = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
START_TAG = "<START>"
STOP_TAG = "<STOP>"

fusion_train = 'data/fusion_data/fusion_train.pth'
fusion_test = 'data/fusion_data/fusion_test.pth'

triger_fusion_train = 'data/fusion_data/triger_fusion_train.pth'
triger_fusion_test = 'data/fusion_data/triger_fusion_test.pth'

argument_fusion_train = 'data/fusion_data/argument_fusion_train.pth'
argument_fusion_test = 'data/fusion_data/argument_fusion_test.pth'

sample_data_path = 'data/sample_data.pth'
test_data_path = 'data/test_data.pth'

keywords_data = 'data/keywords_data/keywords.pth'

tag_2_id = {PAD_WORD: 0, START_TAG: 1, STOP_TAG: 2, 'O': 3, 'B-fusion': 4, 'I-fusion': 5, 'B-part': 6, 'I-part': 7}
word_2_id = {PAD_WORD: 0, START_TAG: 1, STOP_TAG: 2, UNK_WORD: 3}

init_tag_2_id = {PAD_WORD: 0, START_TAG: 1, STOP_TAG: 2}
init_word_2_id = {PAD_WORD: 0, START_TAG: 1, STOP_TAG: 2, UNK_WORD: 3}

msra_data_path = 'data/msra_data.pth'
