# Prerequisites

Python 3.8 or above, suggest using anaconda/conda as the environment manager.

# Install

  1. Run `pip install -r requirements.txt`

# Run

Perform an analysis on a set of data using: `python cofp.py <path_to_directory_containing_pdfs>` more options can be seen by running `python cofp.py --help`.
You can convert the output json into a xlsx file by running `python conv_to_csv.py <path_to_json_file>`

# Output
It is recommended to check the output json to ensure correct text recognization, especially for challenging handwriting. Sublime Text (https://www.sublimetext.com) is a suggested program.
