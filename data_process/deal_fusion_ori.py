import xlrd


def clean_cell(str):
    str = str.strip().split('\n')
    str = '。'.join([s for s in str if s.strip() != ''])
    str = str.replace('\n', '。').replace('【', '').replace('】', '') \
        .replace('　', '。').replace('  	', '。').replace(' ', ''). \
        replace('\t', '').replace('。。。', '。').replace('。。', '')
    return str


book = xlrd.open_workbook('./data/fusion_test.xlsx')
sheet = book.sheet_by_index(0)
with open('./data/fusion_test.txt', 'w') as f:
    for i in sheet.get_rows():
        t1, t2, f1, text = i[0].value, i[1].value, i[2].value, clean_cell(i[3].value)
        f.write(f'{t1}\t{t2}\t{f1}\t{text}\n')
