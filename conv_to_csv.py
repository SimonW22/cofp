import argparse
import os
import json
import xlsxwriter

headers = [
    'ID',
    'Cupper_Name',
    'Cup_Date',
    'Panel_Code',
    'Panel',
    'Sample_Code',
    'Att_Fragrance_1',
    'Roast_Level',
    'Att_Fragrance_2',
    'Roast_Date',
    'Att_Fragrance_3',
    'Att_Fragrance_4',
    'Att_Fragrance_5',
    'Int_Frag_1',
    'Int_Frag_2',
    'Att_Fragrance_6',
    'Int_Frag_3',
    'Int_Frag_4',
    'Int_Frag_5',
    'Att_Aroma_1',
    'Att_Aroma_2',
    'Att_Aroma_3',
    'Att_Aroma_4',
    'Int_Frag_6',
    'Att_Aroma_5',
    'Int_Arom_1',
    'Int_Arom_2',
    'Int_Arom_3',
    'Int_Arom_4',
    'Int_Arom_5',
    'Att_Flavour_1',
    'Att_Flavour_2',
    'Att_Flavour_3',
    'Att_Flavour_4',
    'Att_Flavour_5',
    'Int_Fl_1',
    'Int_Fl_2',
    'Int_Fl_3',
    'Int_Fl_4',
    'Int_Fl_5',
    'Att_Aftertaste_1',
    'Att_Aftertaste_2',
    'Att_Aftertaste_3',
    'Att_Aftertaste_4',
    'Att_Aftertaste_5',
    'Int_Af_1',
    'Int_Af_2',
    'Int_Af_3',
    'Int_Af_4',
    'Int_Af_5',
    'Acidity_1',
    'Acidity_2',
    'Acidity_3',
    'Acidity_4',
    'Acidity_5',
    'Int_Ac_1',
    'Int_Ac_2',
    'Int_Ac_3',
    'Int_Ac_4',
    'Int_Ac_5',
    'Mouthfeel_1',
    'Mouthfeel_2',
    'Mouthfeel_3',
    'Mouthfeel_4',
    'Mouthfeel_5',
    'Int_Mf_1',
    'Int_Mf_2',
    'Int_Mf_3',
    'Int_Mf_4',
    'Int_Mf_5'
]

def write_rows(wksht, row, data):
    for i in range(len(data)):
        wksht.write_string(row, i, data[i])

if __name__ == '__main__':
    # Parse the CLI arguments.
    parser = argparse.ArgumentParser(
        prog='conv_to_csv.py',
        description='Convert json output into csv')
    parser.add_argument('json_input')
    parser.add_argument('--output_path', action='store', help='The output path.')
    args = parser.parse_args()

    if os.path.isabs(args.json_input):
        input_path = args.json_input
    else:
        input_path = os.path.join(os.getcwd(), args.json_input)

    if args.output_path is not None:
        output_path = args.output_path
    else:
        epath, ext = os.path.splitext(input_path)
        output_path = epath + '.xlsx'

    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    if not os.path.exists(input_path):
        print(f'Cannot find input json {input_path}')

    # Load the json and perform the conversion.
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    workbook = xlsxwriter.Workbook(output_path)
    worksheet = workbook.add_worksheet()
    write_rows(worksheet, 0, headers)

    cnt = 1
    for k in input_data.keys():
        for pgs in input_data[k]['SamplePages']:
            for smpls in pgs['Samples']:
                row = [
                    '',
                    input_data[k]['Cupper_Name'],
                    pgs['Cup_Date'],
                    pgs['Panel_Code'],
                    pgs['Panel'],
                    smpls['Sample_Code'],
                    smpls['Att_Fragrance'][0],
                    smpls['Roast_Level'],
                    smpls['Att_Fragrance'][1],
                    smpls['Roast_Date'],
                    smpls['Att_Fragrance'][2],
                    smpls['Att_Fragrance'][3],
                    smpls['Att_Fragrance'][4],
                    smpls['Int_Frag'][0],
                    smpls['Int_Frag'][1],
                    '',
                    smpls['Int_Frag'][2],
                    smpls['Int_Frag'][3],
                    smpls['Int_Frag'][4],
                    smpls['Att_Aroma'][0],
                    smpls['Att_Aroma'][1],
                    smpls['Att_Aroma'][2],
                    smpls['Att_Aroma'][3],
                    '',
                    smpls['Att_Aroma'][4],
                    smpls['Int_Arom'][0],
                    smpls['Int_Arom'][1],
                    smpls['Int_Arom'][2],
                    smpls['Int_Arom'][3],
                    smpls['Int_Arom'][4],
                    smpls['Att_Flavour'][0],
                    smpls['Att_Flavour'][1],
                    smpls['Att_Flavour'][2],
                    smpls['Att_Flavour'][3],
                    smpls['Att_Flavour'][4],
                    smpls['Int_Fl'][0],
                    smpls['Int_Fl'][1],
                    smpls['Int_Fl'][2],
                    smpls['Int_Fl'][3],
                    smpls['Int_Fl'][4],
                    smpls['Att_Aftertaste'][0],
                    smpls['Att_Aftertaste'][1],
                    smpls['Att_Aftertaste'][2],
                    smpls['Att_Aftertaste'][3],
                    smpls['Att_Aftertaste'][4],
                    smpls['Int_Af'][0],
                    smpls['Int_Af'][1],
                    smpls['Int_Af'][2],
                    smpls['Int_Af'][3],
                    smpls['Int_Af'][4],
                    smpls['Acidity'][0],
                    smpls['Acidity'][1],
                    smpls['Acidity'][2],
                    smpls['Acidity'][3],
                    smpls['Acidity'][4],
                    smpls['Int_Ac'][0],
                    smpls['Int_Ac'][1],
                    smpls['Int_Ac'][2],
                    smpls['Int_Ac'][3],
                    smpls['Int_Ac'][4],
                    smpls['Mouthfeel'][0],
                    smpls['Mouthfeel'][1],
                    smpls['Mouthfeel'][2],
                    smpls['Mouthfeel'][3],
                    smpls['Mouthfeel'][4],
                    smpls['Int_Mf'][0],
                    smpls['Int_Mf'][1],
                    smpls['Int_Mf'][2],
                    smpls['Int_Mf'][3],
                    smpls['Int_Mf'][4],
                ]
                write_rows(worksheet, cnt, row)
                cnt += 1
    workbook.close()
