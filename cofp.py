import os
import os.path
import sys
import argparse
import boto3
import botocore.errorfactory
import hashlib
import base64
import json
import time
import pypdf
import PIL
import io
import Levenshtein
import dateutil
import re

_word_re = re.compile(r'[a-z\'\.]+')
_anti_word_re = re.compile(r'[^a-z\'\.]+')

def clean_word(word):
    return word.strip().strip('.,').lower().replace('`', '\'')

def word_list(path):
    found = {}
    with open(path, 'rt') as f:
        for word in f.readlines():
            for sword in word.split():
                cword = clean_word(sword)
                if _anti_word_re.search(cword) is not None or cword in found:
                    continue
                found[cword] = 1
                yield cword

def word_dict(path):
    d = {}
    for word in word_list(path):
        d[word] = 1
    return d

# _words_all = list(word_list(os.path.join(os.path.dirname(__file__), 'words_all.txt')))
_words_all = list(word_list(os.path.join(os.path.dirname(__file__), 'words_simpler.txt')))
_words_blocked = word_dict(os.path.join(os.path.dirname(__file__), 'words_blocked.txt'))
_words_boosted = word_dict(os.path.join(os.path.dirname(__file__), 'words_boosted.txt'))

def correct_text(text):
    global _words_all, _words_blocked, _words_boosted
    corr_words = []
    # For each word in the text, try and match against all dictionary words that
    # have a Levenshtein distance under or equal to 3.
    for word in text.split():
        cword = _anti_word_re.sub('', clean_word(word))
        if len(cword) == 0:
            continue
        matches = []
        # Match against words in dictionary.
        for dword in _words_all:
            r = Levenshtein.distance(dword, cword, score_cutoff=3)
            if r > 3 or dword in _words_blocked:
                continue
            if r == 0:
                matches = [(0, dword)]
                break
            matches += [(r*(0.1 if dword in _words_boosted else 1.0), dword)]
        # Sort matches and take highest value.
        matches.sort(key=lambda x: x[0])
        if len(matches) == 0 or matches[0][0] > 2:
            # print(f'no match {word}')
            corr_words += [word]
        else:
            # print(f'match {word} {matches[0][1]}')
            corr_words += [matches[0][1]]
    return ' '.join(corr_words)

def ask_yesno_question(q):
    while True:
        sys.stdout.write(q + ' [y/n]: ')
        sys.stdout.flush()
        ret = input().lower()
        if ret == 'yes' or ret == 'y':
            return True
        elif ret == 'no' or ret == 'n':
            return False
        else:
            sys.stdout.write('Invalid response, please use either y or n.\n')
            sys.stdout.flush()

class TextractOutput:
    '''
    A class for managing textract outputs.
    '''

    def __init__(self, path : str) -> None:
        with open(path, 'r') as f:
            self._data = json.load(f)
        # Create a ref map for the blocks.
        self._blocks = {}
        for blk in self._data['Blocks']:
            self._blocks[blk['Id']] = blk

    def has_text(self, text):
        for blk in self._data['Blocks']:
            if blk['BlockType'] != 'LINE':
                continue
            if text.lower() == blk['Text'].lower():
                return True
        return False

    def get_block(self, bid):
        return TextractBlock(self, bid)

    def get_form(self):
        return TextractForm(self)

    def get_tables(self):
        for tblk in self.find_blocks_by_type('TABLE'):
            yield TextractTable(self, tblk.id())

    def find_blocks_by_type(self, blockType, entityType=None):
        for k in self._blocks.keys():
            if self._blocks[k]['BlockType'] != blockType:
                continue
            if entityType is not None and entityType not in self._blocks[k]['EntityTypes']:
                continue
            yield TextractBlock(self, k)

class TextractBlock:
    '''
    A class for managing textract blocks.
    '''

    def __init__(self, te : TextractOutput, bid : str) -> None:
        self._te = te
        self._block = self._te._blocks[bid]

    def get(self, key):
        return self._block[key]

    def id(self):
        return self._block['Id']

    def find_rels_of_type(self, ctype, block_type=None):
        if 'Relationships' not in self._block:
            return
        for rel in self._block['Relationships']:
            if rel['Type'] != ctype:
                continue
            for chid in rel['Ids']:
                chblk = self._te.get_block(chid)
                if block_type is not None and block_type != chblk.get('BlockType'):
                    continue
                yield chblk

    def combine_children_values(self, skip_non_text=True):
        txts = []
        for chb in self.find_rels_of_type('CHILD'):
            if chb.get('BlockType') == 'SELECTION_ELEMENT':
                if skip_non_text:
                    continue
                elif chb.get('SelectionStatus') == 'SELECTED':
                    return True
                else:
                    return False
            txts += [chb.get('Text')]
        return ' '.join(txts)

class TextractForm:
    '''
    A class for managing textract forms.
    '''

    def __init__(self, te : TextractOutput) -> None:
        self._te = te
        self._key_value_pairs = []
        for key_block in self._te.find_blocks_by_type('KEY_VALUE_SET', 'KEY'):
            key = key_block.combine_children_values().strip(':').lower()
            val_block = next(key_block.find_rels_of_type('VALUE'))
            value = val_block.combine_children_values(skip_non_text=False)
            self._key_value_pairs += [[key, value]]

    def get_value(self, key : str):
        dup = False
        last_value = None
        last_ratio = 0.0
        for kvp in self._key_value_pairs:
            r = Levenshtein.ratio(kvp[0], key.lower(), score_cutoff=0.9)
            if r == 0.0:
                continue
            if last_value is None:
                last_value = kvp[1]
                last_ratio = r
            elif last_ratio < r:
                last_value = kvp[1]
                last_ratio = r
                dup = True
        return last_value, dup

    def get_date(self, key : str):
        exp_date, dup = self.get_value(key)
        try:
            exp_date = dateutil.parser.parse(exp_date, dayfirst=True).strftime('%d-%b-%y')
        except:
            return None, dup
        return exp_date, dup

    def print_vals(self):
        for kvp in self._key_value_pairs:
            print(f'{kvp[0]}: {kvp[1]}')

class TextractTable(TextractBlock):
    '''
    A class for managing textract tables.
    '''

    def __init__(self, te : TextractOutput, bid : str) -> None:
        super().__init__(te, bid)
        # Extract the table cells.
        self._row_count, self._col_count = 0, 0
        self._rows = {}
        for chblk in self.find_rels_of_type('CHILD', block_type='CELL'):
            row_idx = chblk.get('RowIndex')
            col_idx = chblk.get('ColumnIndex')
            self._row_count = max(self._row_count, row_idx)
            self._col_count = max(self._col_count, col_idx)
            if row_idx not in self._rows:
                self._rows[row_idx] = {}
            self._rows[row_idx][col_idx] = chblk.combine_children_values()
        self._merged_cells = []
        for mcblk in self.find_rels_of_type('MERGED_CELL'):
            min_row, max_row = 10000, -10000
            min_col, max_col = 10000, -10000
            for chblk in mcblk.find_rels_of_type('CHILD', block_type='CELL'):
                row_idx = chblk.get('RowIndex')
                col_idx = chblk.get('ColumnIndex')
                min_row = min(min_row, row_idx)
                max_row = max(max_row, row_idx)
                min_col = min(min_col, col_idx)
                max_col = max(max_col, col_idx)
                self._merged_cells += [[[min_row, max_row], [min_col, max_col]]]

    def check_values(self, row, col, val, nrows=1, ncols=1, default=''):
        v = self.get_values(row, col, nrows=nrows, ncols=ncols, default=default)
        return Levenshtein.ratio(v, val, score_cutoff=0.9) > 0

    def get_values_list(self, row, col, nrows=1, ncols=1, skip_empty=False):
        vals = []
        for i in range(row, row+nrows):
            if i not in self._rows:
                vals += ['']*ncols
                continue
            for j in range(col, col+ncols):
                if j not in self._rows[i]:
                    vals += ['']
                    continue
                v = self._rows[i][j].lower()
                if len(v) == 0 and skip_empty:
                    continue
                vals += [v]
        return vals

    def get_values(self, row, col, nrows=1, ncols=1, default=''):
        vals = self.get_values_list(row, col, nrows=nrows, ncols=ncols, skip_empty=True)
        if len(vals) == 0:
            return default
        return ', '.join(vals)

class CoffeeParser:
    '''
    A coffee parser object used to parse sensory descriptor forms.
    '''

    def __init__(self, data_dir : str, output_path : str, aws_profile : str = 'default', force_unlock : bool = False) -> None:
        '''
        Creates a new coffee parser object.

        Parameters
        ----------
        data_dir : str
            The directory containing pdfs to be analysed. A standard front page form 
            followed by the sample forms is assumed.
        output_path : str
            The path to the json with the calculated values.
        aws_profile : str
            The name of the AWS profile to use, the default is used if none is given.
        force_unlock : bool
            Force the remote s3 state to unlock.
        '''
        
        if type(data_dir) is not str:
            raise TypeError(f'data_dir should be of type string')
        if output_path is not None and type(output_path) is not str:
            raise TypeError(f'output_path should be of type string or None')
        if output_path is None:
            output_path = 'output.json'
        if type(aws_profile) is not str:
            raise TypeError(f'aws_profile should be of type string')
        if aws_profile == '':
            raise ValueError(f'aws_profile should not be empty')

        self._data_dir = os.path.join(os.getcwd(), data_dir) if not os.path.isabs(data_dir) else data_dir
        self._output_path = os.path.join(os.getcwd(), output_path) if not os.path.isabs(output_path) else output_path

        if not os.path.exists(self._data_dir) or not os.path.isdir(self._data_dir):
            raise ValueError(f'data_dir either does not exist or is not a directory')

        self._aws_profile = aws_profile

        self._state_file = os.path.join(self._data_dir, 'state.json')
        self._scan_dir = os.path.join(self._data_dir, 'scans')

    def __enter__(self):
        self._aws_session = boto3.Session(profile_name=self._aws_profile)
        self._textract_client = self._aws_session.client('textract')
        os.makedirs(self._scan_dir, exist_ok=True)
        # Lock the state.
        self._load_state()
        if self._state['lockedAt'] is not None:
            raise Exception('State is currently locked.')
        self._state['lockedAt'] = time.time()
        self._save_state()
        return self

    def __exit__(self, ex_type, exc_val, exc_t):
        # Unlock the state.
        self._state['lockedAt'] = None
        self._save_state()
        return None

    def process(self, force_scan : bool = False) -> None:
        '''
        Submits the pdfs using AWS Textract and then parses the information into the form required.

        Parameters
        ----------
        force_scan : bool
            Re-process pages via AWS Textract.
        '''

        # Check if output exists and ask to overwrite.
        if os.path.exists(self._output_path):
            if not ask_yesno_question('Output already exists, do you wish to overwrite it?'):
                return

        # Scan data directory for PDFs and sync them to the bucket.
        print(f'Processing PDFs in "{self._data_dir}".')
        for fn in os.listdir(self._data_dir):
            # Push to s3.
            fpath = os.path.join(self._data_dir, fn)
            fname, fext = os.path.splitext(fn)
            if os.path.isdir(fpath) or fext.lower() != '.pdf':
                continue

            # Scan using textract.
            if fn not in self._state['files'] or force_scan:
                print(f'  - Found "{fn}" submitting to textract.')
                for i, img_data in self._load_pdf_images(fpath):
                    print(f'    - Processing page {i}.')
                    r = self._textract_client.analyze_document(Document={'Bytes': img_data}, FeatureTypes=['TABLES', 'FORMS'])
                    self._save_json(os.path.join(self._scan_dir, f'{fname}.{i}.json'), r)
                self._state['files'][fn] = time.time()
                self._save_state()

        # Process output from textract.
        parsed_data = {}
        issues = []
        print(f'Processing Textract outputs to "{self._output_path}".')
        for fn in self._state['files']:
            print(f'  - Parsing {fn}')
            pdf_data_extract = {
                'Cupper_Name': '',
                'SamplePages': [],
                '@Issues': []
            }
            # Load the output pages.
            fname, fext = os.path.splitext(fn)
            tx_pages_data = []
            pgcnt = 0
            while True:
                pgpath = os.path.join(self._scan_dir, f'{fname}.{pgcnt}.json')
                if not os.path.exists(pgpath):
                    break
                tx_pages_data += [TextractOutput(pgpath)]
                pgcnt += 1
            # Process the first page.
            sensory_start_page = 0
            if self._is_opening_page(tx_pages_data[0]):
                pdf_data_extract = self._parse_opening_page(pdf_data_extract, tx_pages_data[0])
                sensory_start_page = 1
            # Process subsequent pages.
            for i in range(sensory_start_page,len(tx_pages_data)):
                pdf_data_extract = self._parse_sample_page(pdf_data_extract, tx_pages_data[i])
            parsed_data[fn] = pdf_data_extract
        self._save_json(self._output_path, parsed_data, indent=2)

    def _is_opening_page(self, data):
        return data.has_text('Sensory Descriptor Standardisation Form')

    def _parse_opening_page(self, pdf_data_extract, data):
        issues = []

        form = data.get_form()

        first_name, dup = form.get_value('first name')
        if dup:
            issues += ['possible duplicate first name']
        elif first_name is None:
            first_name = '<Unknown>'

        last_name, dup = form.get_value('last name')
        if dup:
            issues += ['possible duplicate last name']
        elif last_name is None:
            last_name = '<Unknown>'
        
        pdf_data_extract['Cupper_Name'] = f'{first_name} {last_name}'.title()
        pdf_data_extract['@Issues'] = issues

        return pdf_data_extract

    def _parse_sample_page(self, pdf_data_extract, data):
        issues = []

        form = data.get_form()

        if pdf_data_extract['Cupper_Name'] == '':
            name, dup = form.get_value('name')
            if dup:
                issues += ['possible duplicate name']
            if name is not None:
                pdf_data_extract['Cupper_Name'] = name.title()

        # Process the form information.
        cup_date, dup = form.get_date('date')
        if dup:
            issues += ['possible duplicate date']
        elif cup_date is None:
            cup_date = '<Unknown>'

        cupping_panel, dup = form.get_value('cupping panel')
        if dup:
            issues += ['possible duplicate cupping panel']
        elif cupping_panel is None:
            cupping_panel = '<Unknown>'

        # Process the table information.
        tables = list(data.get_tables())
        if len(tables) == 0:
            sample_data = []
            issues += ['found no tables in page']
        else:
            sample_data, sissues = self._extract_sample_table_data(tables[0])
            issues += sissues
        if len(tables) > 1:
            issues += ['found multiple tables in page']

        # Construct the page data.
        page_data = {
            'Cup_Date': cup_date,
            'Panel_Code': '',
            'Panel': cupping_panel,
            'Samples': sample_data,
            '@Issues': issues
        }
        pdf_data_extract['SamplePages'] += [page_data]

        return pdf_data_extract
 
    def _extract_sample_table_data(self, sample_table : TextractTable):
        sample_data = []
        issues = []

        if sample_table._row_count != 31:
            issues += [f'sample table unexpected number of rows {sample_table._row_count}']
        if sample_table._col_count != 13:
            issues += [f'sample table unexpected number of columns {sample_table._col_count}']

        valid_headers = sample_table.check_values(1, 1, 'sample') and \
                        sample_table.check_values(1, 2, 'fragrance (dry)') and \
                        sample_table.check_values(1, 3, 'int') and \
                        sample_table.check_values(1, 4, 'aroma (wet)') and \
                        sample_table.check_values(1, 5, 'int') and \
                        sample_table.check_values(1, 6, 'flavour') and \
                        sample_table.check_values(1, 7, 'int') and \
                        sample_table.check_values(1, 8, 'aftertaste') and \
                        sample_table.check_values(1, 9, 'int') and \
                        sample_table.check_values(1, 10, 'acidity') and \
                        sample_table.check_values(1, 11, 'int') and \
                        sample_table.check_values(1, 12, 'mouthfeel') and \
                        sample_table.check_values(1, 13, 'int')
        if not valid_headers:
            issues += [f'found invalid headers in sample table']

        for n in range(6):
            sample_data += [{
                'Sample_Code': self._clean_sample_no(sample_table.get_values(2+5*n, 1, 5, 1)),
                'Roast_Level': '',  #??
                'Roast_Date': '',   #??

                'Att_Fragrance': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 2, 5, 1)],
                'Int_Frag': sample_table.get_values_list(2+5*n, 3, 5, 1),
                
                'Att_Aroma': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 4, 5, 1)],
                'Int_Arom': sample_table.get_values_list(2+5*n, 5, 5, 1),
                
                'Att_Flavour': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 6, 5, 1)],
                'Int_Fl': sample_table.get_values_list(2+5*n, 7, 5, 1),
                
                'Att_Aftertaste': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 8, 5, 1)],
                'Int_Af': sample_table.get_values_list(2+5*n, 9, 5, 1),
                
                'Acidity': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 10, 5, 1)],
                'Int_Ac': sample_table.get_values_list(2+5*n, 11, 5, 1),
                
                'Mouthfeel': [correct_text(v) for v in sample_table.get_values_list(2+5*n, 12, 5, 1)],
                'Int_Mf': sample_table.get_values_list(2+5*n, 13, 5, 1),
                
            }]

        return sample_data, issues

    def _clean_sample_no(self, sample_no : str) -> str:
        ints = [int(v.strip('.').strip()) for v in sample_no.split(',') if re.match(r'^\s*\d+[\.\s]*$', v)]
        if len(ints) == 0:
            return ''
        ints.sort(reverse=True)
        return str(ints[0])

    def _md5hash(self, path : str) -> str:
        with open(path,'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _save_json(self, path, obj, **kwargs):
        with open(path, 'w') as f:
            json.dump(obj, f, **kwargs)

    def _load_json(self, path):
        if not os.path.exists(self._state_file):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _save_state(self):
        with open(self._state_file, 'w') as f:
            json.dump(self._state, f)

    def _load_state(self):
        if os.path.exists(self._state_file):
            with open(self._state_file, 'r') as f:
                self._state = json.load(f)
        else:
            self._state = {
                'lockedAt': None,
                'files': {}
            }
    
    def _load_pdf_images(self, path):
        '''
        Return a page number and the data for the largest image on the page.

        Parameters
        ----------
        path : str
            Path to the PDF to process.
        '''
        pdf_reader = pypdf.PdfReader(path)
        for page, i in zip(pdf_reader.pages, range(len(pdf_reader.pages))):
            largest_image = None
            for image_file_object in page.images:
                if largest_image is not None and len(largest_image.data) > len(image_file_object.data):
                    continue
                largest_image = image_file_object
            # Make sure the image is a RGB JPEG.
            im = PIL.Image.open(io.BytesIO(largest_image.data))
            with io.BytesIO() as myio:
                im.convert('RGB').save(myio, format='JPEG', quality=80)
                myio.seek(0)
                yield i, myio.read()

if __name__ == '__main__':
    # Parse the CLI arguments.
    parser = argparse.ArgumentParser(
        prog='cofp.py',
        description='Parses sensory descriptor forms')
    parser.add_argument('directory')
    parser.add_argument('--output_path', action='store', help='The output path.')
    parser.add_argument('--aws_profile', action='store', default='default', help='The name of the AWS profile to use.')
    parser.add_argument('--rescan', action='store_true', default=False, help='Rescan the data using AWS Textract.')
    parser.add_argument('--unlock', action='store_true', default=False, help='Force unlocking of state in s3 bucket.')
    args = parser.parse_args()

    # Process the data.
    with CoffeeParser(data_dir=args.directory, output_path=args.output_path, aws_profile=args.aws_profile, force_unlock=args.unlock) as cp:
        cp.process(force_scan=args.rescan)
